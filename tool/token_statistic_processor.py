import json

class TokenStatisticProcessor():

    agent_count = 0
    prompt_tokens = 0
    completiton_tokens = 0
    total_tokens = 0

    @staticmethod
    def clear():
        TokenStatisticProcessor.agent_count = 0
        TokenStatisticProcessor.prompt_tokens = 0
        TokenStatisticProcessor.completiton_tokens = 0
        TokenStatisticProcessor.total_tokens = 0

    @staticmethod
    def dump_to_json(profile_path):
        token_info_dict = {}
        token_info_dict['agent_count'] = TokenStatisticProcessor.agent_count
        token_info_dict['prompt_tokens'] = TokenStatisticProcessor.prompt_tokens
        token_info_dict['completiton_tokens'] = TokenStatisticProcessor.completiton_tokens
        token_info_dict['total_tokens'] = TokenStatisticProcessor.total_tokens
        if TokenStatisticProcessor.agent_count > 0:
            count = TokenStatisticProcessor.agent_count
        else:
            count = 1
        token_info_dict['avg_prompt_tokens'] = TokenStatisticProcessor.prompt_tokens/count
        token_info_dict['avg_completiton_tokens'] = TokenStatisticProcessor.completiton_tokens/count
        token_info_dict['avg_total_tokens'] = TokenStatisticProcessor.total_tokens/count

        with open(profile_path, 'w') as f:
            json.dump(token_info_dict, f, indent=4)

    @staticmethod
    def statistic_token_with_single_agent_from_response(response):
        usage = response.info.get('usage', {})
        prompt_tokens = usage.get('prompt_tokens')
        total_tokens = usage.get('total_tokens')
        completion_tokens = total_tokens - prompt_tokens
        TokenStatisticProcessor.agent_count += 1
        TokenStatisticProcessor.prompt_tokens += prompt_tokens
        TokenStatisticProcessor.total_tokens += completion_tokens
        TokenStatisticProcessor.total_tokens += total_tokens