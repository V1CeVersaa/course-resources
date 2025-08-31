#[derive(Debug, Clone)]
pub enum RespValue {
    SimpleString(String),
    Error(String),
    Integer(i64),
    BulkString(Option<String>), // None represents null
    Array(Option<Vec<RespValue>>),
}

pub struct RespParser {
    data: Vec<u8>,
    position: usize,
}

impl RespParser {
    pub fn new(data: Vec<u8>) -> Self {
        Self { data, position: 0 }
    }

    pub fn parse(&mut self) -> Result<RespValue, String> {
        if self.position >= self.data.len() {
            return Err("Unexpected end of input".to_string());
        }

        let type_char = self.data[self.position];
        self.position += 1;

        match type_char {
            b'+' => self.parse_simple_string(),
            b'-' => self.parse_error(),
            b':' => self.parse_integer(),
            b'$' => self.parse_bulk_string(),
            b'*' => self.parse_array(),
            _ => Err(format!("Unknown type character: {}", type_char as char)),
        }
    }

    fn parse_simple_string(&mut self) -> Result<RespValue, String> {
        let line = self.read_line()?;
        Ok(RespValue::SimpleString(line))
    }

    fn parse_error(&mut self) -> Result<RespValue, String> {
        let line = self.read_line()?;
        Ok(RespValue::Error(line))
    }

    fn parse_integer(&mut self) -> Result<RespValue, String> {
        let line = self.read_line()?;
        let num = line.parse::<i64>().map_err(|_| format!("Invalid integer: {}", line))?;
        Ok(RespValue::Integer(num))
    }

    fn parse_bulk_string(&mut self) -> Result<RespValue, String> {
        let length_str = self.read_line()?;
        let length = length_str
            .parse::<i32>()
            .map_err(|_| format!("Invalid bulk string length: {}", length_str))?;

        if length == -1 {
            return Ok(RespValue::BulkString(None)); // null bulk string
        }

        if length < 0 {
            return Err(format!("Invalid bulk string length: {}", length));
        }

        let content = self.read_bytes(length as usize)?;
        self.expect_crlf()?;

        Ok(RespValue::BulkString(Some(
            String::from_utf8_lossy(&content).to_string(),
        )))
    }

    fn parse_array(&mut self) -> Result<RespValue, String> {
        let count_str = self.read_line()?;
        let count = count_str
            .parse::<i32>()
            .map_err(|_| format!("Invalid array count: {}", count_str))?;

        if count == -1 {
            return Ok(RespValue::Array(None)); // null array
        }

        if count < 0 {
            return Err(format!("Invalid array count: {}", count));
        }

        let mut elements = Vec::new();
        for _ in 0..count {
            let element = self.parse()?;
            elements.push(element);
        }

        Ok(RespValue::Array(Some(elements)))
    }

    fn read_line(&mut self) -> Result<String, String> {
        let mut line = Vec::new();

        while self.position < self.data.len() - 1 {
            if self.data[self.position] == b'\r' && self.data[self.position + 1] == b'\n' {
                self.position += 2; // Skip \r\n
                return Ok(String::from_utf8_lossy(&line).to_string());
            }
            line.push(self.data[self.position]);
            self.position += 1;
        }

        Err("Line not terminated with \\r\\n".to_string())
    }

    fn read_bytes(&mut self, count: usize) -> Result<Vec<u8>, String> {
        if self.position + count > self.data.len() {
            return Err("Not enough bytes to read".to_string());
        }

        let bytes = self.data[self.position..self.position + count].to_vec();
        self.position += count;
        Ok(bytes)
    }

    fn expect_crlf(&mut self) -> Result<(), String> {
        if self.position + 1 >= self.data.len()
            || self.data[self.position] != b'\r'
            || self.data[self.position + 1] != b'\n'
        {
            return Err("Expected \\r\\n".to_string());
        }
        self.position += 2;
        Ok(())
    }
}

impl RespValue {
    pub fn to_string_response(&self) -> String {
        match self {
            RespValue::SimpleString(s) => format!("+{}\r\n", s),
            RespValue::Error(s) => format!("-{}\r\n", s),
            RespValue::Integer(i) => format!(":{}\r\n", i),
            RespValue::BulkString(Some(s)) => format!("${}\r\n{}\r\n", s.len(), s),
            RespValue::BulkString(None) => "$-1\r\n".to_string(),
            RespValue::Array(Some(arr)) => {
                let mut result = format!("*{}\r\n", arr.len());
                for item in arr {
                    result.push_str(&item.to_string_response());
                }
                result
            }
            RespValue::Array(None) => "*-1\r\n".to_string(),
        }
    }
}

// Minimal streaming parse helper: returns (RespValue, consumed_bytes)
pub fn parse_with_consumed(input: &[u8]) -> Result<(RespValue, usize), String> {
    let mut parser = RespParser::new(input.to_vec());
    let value = parser.parse()?;
    Ok((value, parser.position))
}

// Minimal command parse with consumed bytes (command must be an array)
pub fn parse_command_with_consumed(input: &[u8]) -> Result<(Vec<String>, usize), String> {
    let (parsed, consumed) = parse_with_consumed(input)?;
    match parsed {
        RespValue::Array(Some(elements)) => {
            let mut command = Vec::new();
            for element in elements {
                match element {
                    RespValue::BulkString(Some(s)) => command.push(s),
                    RespValue::SimpleString(s) => command.push(s),
                    _ => return Err("Invalid command format".to_string()),
                }
            }
            Ok((command, consumed))
        }
        _ => Err("Command must be an array".to_string()),
    }
}

pub fn parse_command(input: &[u8]) -> Result<Vec<String>, String> {
    let mut parser = RespParser::new(input.to_vec());
    let parsed = parser.parse()?;

    match parsed {
        RespValue::Array(Some(elements)) => {
            let mut command = Vec::new();
            for element in elements {
                match element {
                    RespValue::BulkString(Some(s)) => command.push(s),
                    RespValue::SimpleString(s) => command.push(s),
                    _ => return Err("Invalid command format".to_string()),
                }
            }
            Ok(command)
        }
        _ => Err("Command must be an array".to_string()),
    }
}
