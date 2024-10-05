import json
def extract_json(content, is_list=False):

        l = content.find('{') if not is_list else content.find('[')
        r = content.rfind('}')+1 if not is_list else content.rfind(']')+1
        c = content[l:r]
        return json.loads(c)
    
    
__all__ = ['extract_json']