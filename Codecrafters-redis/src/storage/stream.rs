use std::fmt::{Display, Formatter};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;

// Type aliases for complex types
type StreamEntryData = Vec<(String, Vec<String>)>;
type StreamReadResult = Vec<(String, StreamEntryData)>;
type StreamResponseSender = mpsc::UnboundedSender<Option<StreamReadResult>>;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct StreamId {
    pub timestamp: u64,
    pub sequence: u64,
}

impl StreamId {
    pub fn parse(id_str: &str, last_id: Option<&StreamId>) -> Result<StreamId, String> {
        if id_str == "*" {
            let current_time = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|_| "Failed to get current time".to_string())?
                .as_millis() as u64;

            let sequence = if let Some(last) = last_id {
                if last.timestamp == current_time {
                    last.sequence + 1 // Same timestamp, increment sequence
                } else {
                    0 // New timestamp, start from 0
                }
            } else {
                0 // Empty stream, start from 0
            };

            return Ok(StreamId {
                timestamp: current_time,
                sequence,
            });
        }

        let parts: Vec<&str> = id_str.split('-').collect();
        if parts.len() > 2 {
            return Err("Invalid stream ID format".to_string());
        }

        let timestamp: u64 = parts[0]
            .parse::<u64>()
            .map_err(|_| "Invalid timestamp in stream ID".to_string())?;

        // Auto-generate sequence number
        let sequence: u64 = if parts[1] == "*" {
            if let Some(last) = last_id {
                if last.timestamp == timestamp {
                    last.sequence + 1 // Same timestamp, increment sequence
                } else if timestamp == 0 {
                    1
                } else {
                    0 // New timestamp, start from 0 (or 1 for timestamp 0)
                }
            } else if timestamp == 0 {
                1
            } else {
                0
            }
        } else {
            parts[1]
                .parse::<u64>()
                .map_err(|_| "Invalid sequence in stream ID".to_string())?
        };

        Ok(StreamId { timestamp, sequence })
    }

    pub fn parse_range_id(id_str: &str, default_sequence: u64) -> Result<StreamId, String> {
        if id_str == "-" {
            Ok(StreamId {
                timestamp: 0,
                sequence: 0,
            })
        } else if id_str == "+" {
            Ok(StreamId {
                timestamp: u64::MAX,
                sequence: u64::MAX,
            })
        } else if id_str.contains('-') {
            let parts: Vec<&str> = id_str.split('-').collect();
            if parts.len() != 2 {
                return Err("Invalid stream ID format".to_string());
            }

            let timestamp: u64 = parts[0]
                .parse::<u64>()
                .map_err(|_| "Invalid timestamp in stream ID".to_string())?;
            let sequence: u64 = parts[1]
                .parse::<u64>()
                .map_err(|_| "Invalid sequence in stream ID".to_string())?;
            Ok(StreamId { timestamp, sequence })
        } else {
            let timestamp: u64 = id_str
                .parse::<u64>()
                .map_err(|_| "Invalid timestamp in stream ID".to_string())?;
            Ok(StreamId {
                timestamp,
                sequence: default_sequence,
            })
        }
    }

    pub fn is_in_range(&self, start: &StreamId, end: &StreamId) -> bool {
        *self >= *start && *self <= *end
    }

    pub fn is_greater_than_zero(&self) -> bool {
        self.timestamp > 0 || self.sequence > 0
    }
}

impl Display for StreamId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}-{}", self.timestamp, self.sequence)
    }
}

#[derive(Debug, Clone)]
pub struct StreamEntry {
    pub id: StreamId,
    pub fields: Vec<(String, String)>,
}

#[derive(Debug)]
pub struct BlockedStreamClient {
    pub client_id: u64,
    pub streams: Vec<(String, String)>, // (key, start_id)
    pub timeout: Option<Duration>,
    pub start_time: Instant,
    pub response_sender: StreamResponseSender,
}

impl Clone for BlockedStreamClient {
    fn clone(&self) -> Self {
        Self {
            client_id: self.client_id,
            streams: self.streams.clone(),
            timeout: self.timeout,
            start_time: self.start_time,
            response_sender: self.response_sender.clone(),
        }
    }
}

pub trait StreamStorage: Send + Sync {
    fn xadd(&mut self, key: String, id: String, fields: Vec<(String, String)>) -> Result<String, String>;
    fn xrange(&self, key: &str, start: &str, end: &str) -> Result<Vec<(String, Vec<String>)>, String>;
    fn xread(&self, streams: Vec<(String, String)>) -> Result<StreamReadResult, String>;
    fn get_max_stream_id(&self, key: &str) -> Option<String>;

    fn add_blocked_stream_client(&mut self, client: BlockedStreamClient);
    fn notify_blocked_stream_clients(&mut self, key: &str);
    fn generate_stream_client_id(&mut self) -> u64;
    fn cleanup_blocked_stream_client(&mut self, client_id: u64, streams: &[(String, String)]);
}
