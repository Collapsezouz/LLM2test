

def byte_safe_decode(data:bytes, encoding='utf8'):
    if data is None:
        return None, None
    
    for i in range(3):
        trim_data = data[:-i] if i else data
        try:
            text = trim_data.decode(encoding=encoding)
            return text, data[-i:] if i else b''
        except UnicodeDecodeError:
            continue