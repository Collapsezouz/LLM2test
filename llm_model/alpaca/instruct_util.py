
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