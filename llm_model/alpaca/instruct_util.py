
def instruct_encode(instruct:dict, version=2):
    if version == 1:
        return instruct_encode_v1(instruct)
    else:
        return instruct_encode_v2(instruct)


def instruct_encode_v1(instruct:dict):
    _input = instruct.get("input", "")
    if _input in ('<noinput>',): _input = ''
    _system, _instruction = instruct.get("system", ""), instruct.get("instruction", "")
    prompt = _system + "\n" if _system else ''
    if _input:
        return prompt + "{instruction}\n\n- 输入:\n{input}\n\n- 输出:".format(
            instruction=_instruction, input= _input)
    else:
        return prompt + "{instruction}\n\n- 输出:".format(
            instruction=_instruction)


def instruct_encode_v2(instruct:dict):
    _input = instruct.get("input", "")
    if _input in ('<noinput>',): _input = ''
    _system, _instruction = instruct.get("system", ""), instruct.get("instruction", "")
    prompt = ''
    if _system:
        prompt += _system + "\n"
    prompt += "<!User>:\n" + _instruction + "\n"
    if _input:
        prompt += '<!Input>:\n' + _input + "\n"
    prompt += '<!Machine>:\n'
    return prompt

def instruct_train_dataset(instruct:dict, version=2):
    _ds = instruct.get('_ds')
    if _ds is None:
        input_text = instruct_encode(instruct, version=version)
        output_text = instruct.get('output')
        yield input_text, output_text
        return
    _system, _instruction = instruct.get("system", ""), instruct.get("instruction", "")
    for fields in _ds:
        input_fields = fields.get('input')
        output_fields = fields.get('output')
        input_text, output_text = '', ''
        # input_text
        _input, _output= '', ''
        for name in input_fields:
            if name == 'system' or name == 'instruction':
                continue
            _input += '<!%s>:\n'%name.capitalize() + instruct[name] + "\n"
        if _system:
            input_text += _system + "\n"
        input_text += "<!User>:\n" + _instruction + "\n"
        input_text += _input #+ "\n"
        # output_text
        for name in output_fields:
            if output_text: output_text += '\n'
            _output_prefix = '<!Call>:\n' if name in ('call', 'query') else '<!Machine>:\n'
            output_text += _output_prefix + instruct[name]
        yield input_text, output_text
