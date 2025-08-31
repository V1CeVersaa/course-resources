use std::time::Duration;
use tokio::sync::mpsc;

#[derive(Debug)]
pub struct BlockedClient {
    pub client_id: u64,
    pub list_keys: Vec<String>,
    pub timeout: Option<Duration>,
    pub start_time: std::time::Instant,
    pub response_sender: mpsc::UnboundedSender<Option<(String, String)>>,
}

impl Clone for BlockedClient {
    fn clone(&self) -> Self {
        Self {
            client_id: self.client_id,
            list_keys: self.list_keys.clone(),
            timeout: self.timeout,
            start_time: self.start_time,
            response_sender: self.response_sender.clone(),
        }
    }
}

pub trait ListStorage: Send + Sync {
    fn rpush(&mut self, key: String, value: String) -> Result<i64, String>;
    fn lpush(&mut self, key: String, value: String) -> Result<i64, String>;
    fn lrange(&self, key: &str, start: i64, stop: i64) -> Vec<String>;
    fn llen(&self, key: &str) -> i64;
    fn lpop(&mut self, key: &str, count: Option<usize>) -> Vec<String>;

    fn try_blpop(&mut self, keys: &[String]) -> Option<(String, String)>;
    fn add_blocked_client(&mut self, client: BlockedClient);
    fn notify_blocked_clients(&mut self, key: &str);
    fn generate_client_id(&mut self) -> u64;
    fn cleanup_blocked_client(&mut self, client_id: u64, keys: &[String]);
}
