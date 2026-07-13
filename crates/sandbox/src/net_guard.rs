//! SSRF-filtered network guard.
//!
//! Pipeline: parse URL → scheme must be https → host must suffix-match allowlist
//! → resolve DNS → reject if any IP is loopback/private/link-local/IPv4-mapped
//! → connect via reqwest with pinned IP (defeats DNS rebinding). No automatic
//! redirects; follow manually up to 3 hops, re-running full checks per hop.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

use agent_types::{AgentError, Result};
use ipnet::{Ipv4Net, Ipv6Net};
use url::Url;

pub struct NetGuard {
    allowed_domains: Vec<String>,
    client: reqwest::Client,
}

impl NetGuard {
    pub fn new(allowed_domains: Vec<String>) -> Self {
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self {
            allowed_domains,
            client,
        }
    }

    /// Fetch a URL with full SSRF protection.
    pub async fn get(&self, url_str: &str) -> Result<String> {
        let mut current_url = url_str.to_string();
        let mut hops = 0;

        loop {
            let url = Url::parse(&current_url).map_err(|e| {
                AgentError::Sandbox(format!("invalid url: {e}"))
            })?;

            // 1. Scheme must be https.
            if url.scheme() != "https" {
                return Err(AgentError::Sandbox(format!(
                    "only https allowed, got '{}'",
                    url.scheme()
                )));
            }

            // 2. Host must suffix-match allowlist.
            let host = url
                .host_str()
                .ok_or_else(|| AgentError::Sandbox("no host in url".into()))?;
            if !self.is_allowed_host(host) {
                return Err(AgentError::Sandbox(format!(
                    "host '{host}' not in allowlist"
                )));
            }

            // 3. Resolve DNS and check all IPs.
            let port = url.port_or_known_default().unwrap_or(443);
            let addrs = resolve_host(host, port).await?;
            for addr in &addrs {
                if is_private_ip(*addr) {
                    return Err(AgentError::Sandbox(format!(
                        "resolved IP {addr} is private/loopback/link-local"
                    )));
                }
            }

            // 4. Make request, PINNING the validated IP so reqwest does NOT
            //    re-resolve DNS (defeats DNS-rebinding TOCTOU). We build a
            //    request-scoped client that maps host→validated addr.
            let pinned_addr = std::net::SocketAddr::new(addrs[0], port);
            let pinned_client = reqwest::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .resolve(host, pinned_addr)
                .build()
                .map_err(|e| AgentError::Sandbox(format!("client build: {e}")))?;

            let resp = pinned_client
                .get(url.as_str())
                .send()
                .await
                .map_err(|e| AgentError::Sandbox(format!("request failed: {e}")))?;

            // 5. Handle redirects manually.
            if resp.status().is_redirection() {
                hops += 1;
                if hops > 3 {
                    return Err(AgentError::Sandbox(
                        "too many redirects (max 3)".into(),
                    ));
                }
                if let Some(location) = resp.headers().get("location") {
                    current_url = location
                        .to_str()
                        .map_err(|_| AgentError::Sandbox("invalid redirect location".into()))?
                        .to_string();
                    // Resolve relative redirects against current URL.
                    if !current_url.starts_with("http") {
                        let base = Url::parse(url.as_str()).unwrap();
                        current_url = base
                            .join(&current_url)
                            .map_err(|e| AgentError::Sandbox(format!("redirect resolve: {e}")))?
                            .to_string();
                    }
                    continue;
                } else {
                    return Err(AgentError::Sandbox("redirect without location".into()));
                }
            }

            if !resp.status().is_success() {
                return Err(AgentError::Sandbox(format!(
                    "http {}",
                    resp.status()
                )));
            }

            let body = resp
                .text()
                .await
                .map_err(|e| AgentError::Sandbox(format!("read body: {e}")))?;
            return Ok(body);
        }
    }

    fn is_allowed_host(&self, host: &str) -> bool {
        let host_lower = host.to_lowercase();
        self.allowed_domains.iter().any(|domain| {
            let d = domain.to_lowercase();
            host_lower == d || host_lower.ends_with(&format!(".{d}"))
        })
    }
}

/// Resolve a hostname to IP addresses.
async fn resolve_host(host: &str, port: u16) -> Result<Vec<IpAddr>> {
    let addr_str = format!("{host}:{port}");
    let addrs: Vec<IpAddr> = tokio::net::lookup_host(&addr_str)
        .await
        .map_err(|e| AgentError::Sandbox(format!("dns resolve '{host}': {e}")))?
        .map(|sa| sa.ip())
        .collect();
    if addrs.is_empty() {
        return Err(AgentError::Sandbox(format!("no IPs for '{host}'")));
    }
    Ok(addrs)
}

/// Check if an IP is private, loopback, link-local, or IPv4-mapped IPv6.
fn is_private_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_private_v4(v4),
        IpAddr::V6(v6) => {
            // Check IPv4-mapped IPv6 (::ffff:x.x.x.x).
            if let Some(mapped) = v6.to_ipv4_mapped() {
                return is_private_v4(mapped);
            }
            is_private_v6(v6)
        }
    }
}

fn is_private_v4(ip: Ipv4Addr) -> bool {
    ip.is_loopback()
        || ip.is_private()
        || ip.is_link_local()
        || is_in_net_v4(ip, "169.254.0.0/16") // link-local / cloud metadata
        || is_in_net_v4(ip, "127.0.0.0/8")
        || is_in_net_v4(ip, "10.0.0.0/8")
        || is_in_net_v4(ip, "172.16.0.0/12")
        || is_in_net_v4(ip, "192.168.0.0/16")
        || is_in_net_v4(ip, "0.0.0.0/8")
}

fn is_private_v6(ip: Ipv6Addr) -> bool {
    ip.is_loopback()
        || is_in_net_v6(ip, "fc00::/7") // unique local
        || is_in_net_v6(ip, "fe80::/10") // link-local
        || is_in_net_v6(ip, "::1/128")
}

fn is_in_net_v4(ip: Ipv4Addr, cidr: &str) -> bool {
    cidr.parse::<Ipv4Net>()
        .map(|net| net.contains(&ip))
        .unwrap_or(false)
}

fn is_in_net_v6(ip: Ipv6Addr, cidr: &str) -> bool {
    cidr.parse::<Ipv6Net>()
        .map(|net| net.contains(&ip))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn private_ips_detected() {
        assert!(is_private_ip(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1))));
        assert!(is_private_ip(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1))));
        assert!(is_private_ip(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1))));
        assert!(is_private_ip(IpAddr::V4(Ipv4Addr::new(172, 16, 0, 1))));
        assert!(is_private_ip(IpAddr::V4(Ipv4Addr::new(169, 254, 169, 254))));
    }

    #[test]
    fn public_ips_allowed() {
        assert!(!is_private_ip(IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8))));
        assert!(!is_private_ip(IpAddr::V4(Ipv4Addr::new(1, 1, 1, 1))));
    }

    #[test]
    fn ipv4_mapped_v6_detected() {
        // ::ffff:169.254.169.254
        let mapped = IpAddr::V6(Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0xa9fe, 0xa9fe));
        assert!(is_private_ip(mapped));
    }

    #[test]
    fn allowlist_suffix_matching() {
        let guard = NetGuard::new(vec!["crates.io".into(), "github.com".into()]);
        assert!(guard.is_allowed_host("crates.io"));
        assert!(guard.is_allowed_host("static.crates.io"));
        assert!(guard.is_allowed_host("api.github.com"));
        assert!(!guard.is_allowed_host("evil.com"));
        assert!(!guard.is_allowed_host("notcrates.io.evil.com"));
    }

    #[tokio::test]
    async fn http_scheme_rejected() {
        let guard = NetGuard::new(vec!["example.com".into()]);
        let result = guard.get("http://example.com").await;
        assert!(matches!(result, Err(AgentError::Sandbox(ref s)) if s.contains("https")));
    }

    #[tokio::test]
    async fn unlisted_host_rejected() {
        let guard = NetGuard::new(vec!["crates.io".into()]);
        let result = guard.get("https://evil.com/malware").await;
        assert!(matches!(result, Err(AgentError::Sandbox(ref s)) if s.contains("allowlist")));
    }
}
