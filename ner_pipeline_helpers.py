import string

def punc_only(strs):
    noktalama = set(string.punctuation)
    return all(char in noktalama for char in strs.replace(" ", ""))
    
def remove_punc(strs):
    noktalama = set(''.join([punc for punc in string.punctuation if punc not in '-_']))
    return  strs.strip(''.join(noktalama))
    
def remove_smaller_lists(lists):
    to_remove = set()

    for i in range(len(lists)):
        for j in range(i + 1, len(lists)):
            if set(lists[i]) & set(lists[j]):
                if len(lists[i]) < len(lists[j]):
                    to_remove.add(j)
                else:
                    to_remove.add(i)

    filtered_lists = [lists[i] for i in range(len(lists)) if i not in to_remove]

    return filtered_lists
    
    
def eliminate_biggers(list_of_entity_adresses):
    clean_one = []
    for liste in list_of_entity_adresses:
        if len(liste) > 5:
            continue
        else:
            clean_one.append(liste)
    return clean_one
    
    

def keep_smaller(items):
    unique_texts = {}

    for item in items:
        is_similar_found = False
        for base_text in list(unique_texts.keys()):
            if item.text.startswith(base_text) or base_text.startswith(item.text):
                is_similar_found = True
                if len(item.text) < len(unique_texts[base_text].text):
                    unique_texts[base_text] = item
                break
        if not is_similar_found:
            unique_texts[item.text] = item
    return list(unique_texts.values())

    
    
def match_case(a, b):
    result = []
    for char_a, char_b in zip(a, b):
        if char_b.isupper():
            result.append(char_a.upper())
        else:
            result.append(char_a.lower())
    # Ekstra karakterler varsa a'nın kalanını ekle
    if len(a) > len(b):
        result.append(a[len(b):])
    return ''.join(result)
    
def singular_preds(absa_raw_pred):
    opts = list(set([i['span'] for i in absa_raw_pred]))
    rnw = []
    for option in opts:
        rnw.append([i for i in absa_raw_pred if i['span'] == option][0])
    return rnw