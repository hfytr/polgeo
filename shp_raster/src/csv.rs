use std::fs::File;
use std::io::prelude::*;
use std::iter::zip;

pub fn write_csv(
    path: &str,
    keys: &Vec<String>,
    data: &Vec<Vec<f64>>,
    ids: &Vec<String>,
) -> std::io::Result<()> {
    let keys_string = keys
        .iter()
        .fold(String::new(), |acc, elem| format!("{acc}{elem}"));
    let mut file_string = format!("id, {keys_string}\n");
    for (row, id) in zip(data.iter(), ids.iter()) {
        let row_data = row
            .iter()
            .fold(String::new(), |acc, elem| format!("{acc}, {elem}"));
        file_string.push_str(&format!("{id}, {row_data}"));
        file_string.push('\n');
    }
    let mut file = File::create(path)?;
    file.write_all(file_string.as_bytes())?;
    Ok(())
}
