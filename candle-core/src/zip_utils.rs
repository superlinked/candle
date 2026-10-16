use std::collections::HashMap;
use std::io::{Read, Seek};
use zip::result::{ZipError, ZipResult};
use zip::ZipArchive;

pub(crate) fn index_for_name<R: Read + Seek>(
    archive: &ZipArchive<R>,
    name: &str,
) -> ZipResult<usize> {
    if name.is_ascii() {
        return archive.index_for_name(name).ok_or(ZipError::FileNotFound);
    }
    let metadata = archive.metadata();
    let mut selected = None;
    for index in 0..archive.len() {
        let entry = metadata.entry(index)?;
        if entry.name()? == name {
            let offset = entry.central_header_start();
            if selected.is_none_or(|(_, previous_offset)| offset > previous_offset) {
                selected = Some((index, offset));
            }
        }
    }
    selected
        .map(|(index, _)| index)
        .ok_or(ZipError::FileNotFound)
}

pub(crate) fn decoded_indices<R: Read + Seek>(archive: &ZipArchive<R>) -> ZipResult<Vec<usize>> {
    let metadata = archive.metadata();
    let mut position_per_name = HashMap::<String, usize>::new();
    let mut indices = Vec::new();
    for index in 0..archive.len() {
        let entry = metadata.entry(index)?;
        let name = entry.name()?;
        if name.is_ascii() {
            indices.push(index);
            continue;
        }
        if let Some(&position) = position_per_name.get(name.as_ref()) {
            if entry.central_header_start()
                > metadata.entry(indices[position])?.central_header_start()
            {
                indices[position] = index;
            }
        } else {
            position_per_name.insert(name.into_owned(), indices.len());
            indices.push(index);
        }
    }
    Ok(indices)
}
