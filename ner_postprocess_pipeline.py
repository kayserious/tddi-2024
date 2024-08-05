from ner_pipeline_helpers import * 
from nameless_entity_extractor import *

import itertools
from spacy.language import Language
from spacy.tokens import Span,Token
from spacy.tokens.doc import Doc

@Language.component("post_process_ner")      
def post_process_ner(doc):
    '''spaCy pipelina ner sonrasina gelecek sekilde baglanir'''
    
    entity_names = [i for i in doc.ents if i.label_ not in ['DATE','TIME','CARDINAL','ORDINAL','MONEY','GPE','PERCENT','QUANTITY','PER']]
    
    nameless_ = get_nameless_entity(doc)
    
    if nameless_ is not None:
        
        clear_ones = []
        
        nameless_indices = [kko.i for kko in nameless_]
        
        for single_entity in entity_names:

            named_indices = [kko.i for kko in single_entity]
            
            if len(list(set(nameless_indices) & set(named_indices))) > 0:
                continue
            else:
                clear_ones.append(single_entity)
        entity_names = clear_ones.copy()
    
    
    entity_adresses = []
    for single_entity in entity_names:
        entity_adress = []
        for entity_component in single_entity:
            entity_adress.append([i.text for i in doc].index(entity_component.text))
        entity_adresses.append(entity_adress)
        
    entity_adresses = eliminate_biggers(entity_adresses)
        
    nameless_entity_adresses = []
    if nameless_ is not None:
        nameless_entity_adress = []
        for entity_component in nameless_:
            nameless_entity_adress.append([i.text for i in doc].index(entity_component.text))
        nameless_entity_adresses.append(nameless_entity_adress)
    
    
    entity_adresses = [list(item) for item in set(tuple(row) for row in entity_adresses)]
    entity_adresses = [list(range(min(i),max(i) + 1)) for i in entity_adresses]
    entity_adresses = remove_smaller_lists(entity_adresses)
    
    nameless_entity_adresses = [list(item) for item in set(tuple(row) for row in nameless_entity_adresses)]
    nameless_entity_adresses = [list(range(min(i),max(i) + 1)) for i in nameless_entity_adresses]
    nameless_entity_adresses = remove_smaller_lists(nameless_entity_adresses)
    
    #entity_adresses = [list(item) for item in set(tuple(row) for row in entity_adresses)]
    
    
    def clear_token(token__):
        
        kaynastirma_ekleri = ['sı','si']
        kaynastirma_map = [i in token__.text for i in kaynastirma_ekleri]
        is_kaynastirma = True if any(kaynastirma_map) else False
        
        original_text = str(token__.text)
        
        if is_kaynastirma:
            
            found_kaynastirma = [kaynastirma_eki for kaynastirma_eki,kaynastirma_eki_logic in zip(kaynastirma_ekleri,kaynastirma_map) if kaynastirma_eki_logic][0]
            
            
        
        token_text = token__.lemma_ if token__.lemma_ != '' else token__.text
        token_text = match_case(token_text,original_text)
        #token_text = token__.text
        apostro_idx = token_text.find("'")
        
        if apostro_idx == -1:
            apostro_idx = token_text.find("’")

        if apostro_idx != -1:
            after_apostro = token_text[apostro_idx+1:]
            cut_len = len(after_apostro)
            to_return = token_text[:-cut_len-1]
            
            if to_return == '':
                return '.'
            else:
                if is_kaynastirma and found_kaynastirma not in to_return:
                    return f'{remove_punc(to_return)}{found_kaynastirma}'
                else:
                    return remove_punc(to_return)
        else:
            to_return = remove_punc(token_text)
            if to_return == '':
                return '.'
            else:
                if is_kaynastirma and found_kaynastirma not in to_return:
                    return f'{to_return}{found_kaynastirma}'
                else:
                    return to_return
            
            
    def clear_nameless_token(token__):
        
        cogul_ekleri = ['lar','ler']
        
        kaynastirma_ekleri = ['sı','si']
        
        original_text = str(token__.text)
        
        plural_map = [i in token__.text for i in cogul_ekleri]
        
        kaynastirma_map = [i in token__.text for i in kaynastirma_ekleri]
        
        is_plural = True if any(plural_map) else False
        
        is_kaynastirma = True if any(kaynastirma_map) else False
        
        if is_plural:
            
            found_plural = [cogul_eki for cogul_eki,cogul_eki_logic in zip(cogul_ekleri,plural_map) if cogul_eki_logic][0]
                
        if is_kaynastirma:
            
            found_kaynastirma = [kaynastirma_eki for kaynastirma_eki,kaynastirma_eki_logic in zip(kaynastirma_ekleri,kaynastirma_map) if kaynastirma_eki_logic][0]
        
        if is_plural or is_kaynastirma:
            
            lemmatized = token__.lemma_
            
            if is_plural and found_plural not in lemmatized:
                
                to_return = f'{lemmatized}{found_plural}'
                to_return = match_case(to_return,original_text)
                
            elif is_kaynastirma and found_kaynastirma not in lemmatized:
                
                to_return = f'{lemmatized}{found_kaynastirma}'
                to_return = match_case(to_return,original_text)
                
            else:
                
                to_return = match_case(lemmatized,original_text)
        else:
            
            to_return = original_text
            
        return to_return
        
    
    words = [clear_token(tok) if tok_id in itertools.chain.from_iterable(entity_adresses) else clear_nameless_token(tok) if tok_id in itertools.chain.from_iterable(nameless_entity_adresses) else tok.text for tok_id,tok in enumerate(doc)]
    spaces = [True if tok.whitespace_ else False for tok in doc]
    
    pos_list = ['PROPN' if xx[0] in itertools.chain.from_iterable(entity_adresses + nameless_entity_adresses) else '' for xx in enumerate(words)]
    
    #print(pos_list)
    
    doc2 = Doc(doc.vocab, words=words, spaces=spaces,pos=pos_list)
    
    new_entities = []
    
    for single_adress in entity_adresses + nameless_entity_adresses:
        adress_begins = min(single_adress)
        adress_ends = max(single_adress)
        newspn = Span(doc2, adress_begins, adress_ends + 1, label='ORG')
        new_entities.append(newspn)
        
    tersliste = new_entities[::-1]
    
    secenekler = list(set([i.text for i in new_entities]))
    buldum = []
    for i in secenekler:
        buldumsayisi = 0
        for j in tersliste:
            if j.text == i and buldumsayisi == 0:
                buldum.append(j)
                buldumsayisi = 1
    #print(secenekler)
    #print(entity_adresses)
    final_entities = buldum[::-1]
    
    final_entities = keep_smaller(final_entities)
    
    final_entities = [ent_ for ent_ in final_entities if not punc_only(ent_.text)]
    doc2.ents = final_entities
    
    for revize in doc2:
        revize.pos_ = ''
    
    for final_ent in doc2.ents:
        
        start_ = final_ent.start
        end_ = final_ent.end
        
        for rn_ in range(start_,end_):
            doc2[rn_].pos_ = 'PROPN'
    
    return doc2