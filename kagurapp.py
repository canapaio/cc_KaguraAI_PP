from cat.mad_hatter.decorators import tool, hook, plugin
from cat.factory.custom_llm import CustomOllama
#from cat.factory.llm import LLMSettings
from pydantic import BaseModel
from datetime import datetime, date
from cat.log import log
import os, re, copy

#############################
@hook
def before_cat_sends_message(message, cat):
    settings = cat.mad_hatter.get_plugin().load_settings()
    # Variabili
    kmp_f: str = settings["kpp_path"] + settings["kpp_mindprefix"] # File prefix mappa mentale
    kmr_f: str = settings["kpp_path"] + "klastmind.txt" # File salvataggio mappa mentale
    kpp_f: str = settings["kpp_path"] + settings["kpp_file"] # File promptprefix Kagura

    #caricamento kagura prompt prefix
    if os.path.exists(kpp_f):
        with open(kpp_f, 'r') as f:
            prefix = f.read()
    else:
        prefix = settings["prompt_prefix"]


    # Carica il prompt prefix di mindprefix
    if os.path.exists(kmp_f):
        with open(kmp_f, 'r') as f:
            kmindprefix = f.read()
    else:
        kmindprefix = "Sei Kagura: Crea una mappa mentale della situazione"

    # Carica il pensiero precedente
    if os.path.exists(kmr_f):
        with open(kmr_f, 'r') as f:
            klastmind = f.read()
    else:
        klastmind = "indaffarata"

    # elabora il prompt del prossimo pensiero
    kmindprefix = f"""
Personaggio di Kagura che sta pensando:
{prefix}
<stato_mentale_dinamico_precedente>
    {kre(klastmind)}
</stato_mentale_dinamico_precedente>
<Discussione>
{kre(cat.stringify_chat_history(latest_n=4))}
</Discussione>
<prompt>
{kmindprefix}
</prompt>
"""
#Sei Kagura e stai pensado: 
#{kmindprefix}

    #cat.send_chat_message(repr(kmindprefix))
    #xyzk = kppdebug(kmindprefix)

    # Elaborazione mentale LLM
    log.info("======================================================")
    if settings['kpp_debug']:
        log.info(kmindprefix)

    # chiamata ad un modello semplificato
    llm_tmp = copy.deepcopy(cat._llm)
    alt_llm = cat.mad_hatter.get_plugin().load_settings().get('num_ctx', settings['kpp_ctx_S'])
    if alt_llm != '':
        llm_tmp.num_ctx = alt_llm
    alt_llm = cat.mad_hatter.get_plugin().load_settings().get('model', settings['kpp_model_s'])
    if alt_llm != '':
        llm_tmp.model = alt_llm

    kmind: str = (llm_tmp.invoke(kmindprefix).content)

#debug
    if settings['kpp_debug']:
        cat.send_chat_message(kre(kmind))
    #xyzk = kppdebug(kmindprefix)

    # Salvataggio pensiero
    with open(kmr_f, 'w') as f:
        f.write(kmind)
    return messagge

############################# Da riscrivere
'''
@hook
def cat_recall_query(user_message, cat):
    settings = cat.mad_hatter.get_plugin().load_settings()
    #kpp_ctx_S = settiongs['']

    kprompt = f"""
Analizza la discussione contenuta in 'testo-da-analizzare' e genera una liste di parole chiavi congroue in italiano ed iglese seguendo le indicazioni contenute in 'regole':
<regole>
- Genera parole chiave in base al 'testo-da-analizzare' in italiano e inglese
- Elenca tutti i termini specifici dal 'testo-da-analizzare'
- Aggiungi chiavi non presenti che siano congrue con l'argomento
- NON COMMENTARE L'ELENCO
- Crea un elecon pulito privo di commenti
</regole>

<testo-da-analizzare>

{kre(cat.stringify_chat_history(latest_n=10))}

</testo-da-analizzare>
PS (crea solo un elenco di parole chiave in base alla sezione 'testo-da-analizzare' usando le 'regole' come indicato sopra )

"""

    #CustomOllama = (base_url='http://192.168.10.10:11434', model='qwen2.5-coder:latest', num_ctx=1024, repeat_last_n=64, repeat_penalty=1.1, temperature=0.8)
    # chiamata ad un modello semplificato
    #log.info("======================================================")
    #log.info(kprompt)
    llm_tmp = copy.deepcopy(cat._llm)
    alt_llm = cat.mad_hatter.get_plugin().load_settings().get('num_ctx', settings['kpp_ctx_S'])
    if alt_llm != '':
        llm_tmp.num_ctx = alt_llm
    alt_llm = cat.mad_hatter.get_plugin().load_settings().get('model', settings['kpp_model_s'])
    if alt_llm != '':
        llm_tmp.model = alt_llm
    #cat.send_chat_message(repr(llm_tmp))
    kpp_qwery = f"""
{kre(llm_tmp.invoke(kprompt).content)}
{kre(cat.stringify_chat_history(latest_n=4))}
"""

    #log.info("======================================================")
    #log.info(kpp_qwery)
    #log.info("======================================================")

    return kpp_qwery
'''

@hook
def agent_prompt_prefix(prefix: str, cat):
    settings = cat.mad_hatter.get_plugin().load_settings()
    k_ppf: str = settings["kpp_path"] + settings["kpp_file"]
    if os.path.exists(k_ppf):
        with open(k_ppf, 'r') as f:
            prefix = "Sei Kagura: /n <Kagura_prompt_prefix>" + f.read()
    else:
        prefix = "Sei Kagura: /n <Kagura_prompt_prefix>" + settings["prompt_prefix"]
    # Carica il pensiero precedente
    kmr_f: str = settings["kpp_path"] + "klastmind.txt"
    if os.path.exists(kmr_f):
        with open(kmr_f, 'r') as f:
            klastmind = f.read()
    prefix += f"""
  <stato_mentale_dinamico>
    {kre(klastmind)}
  <stato_mentale_dinamico>
</Kagura_prompt_prefix>
"""

    return prefix

@hook
def agent_prompt_suffix(suffix, cat):
    settings = cat.mad_hatter.get_plugin().load_settings()

#    suffix = f"""
#<Conversazione>
#    {kre(cat.stringify_chat_history(latest_n=5))}
#</conversaione>"""

    suffix = """<Kagura_suffix>
1. Da qui inizia l'oblio, (conversazioni passate e memoria richiamata dall'embedder) cerca di seguire il contesto e prendi in considerazione solo i dati utili alla discussione e alla tua personalità:
2. La tag "Human" non identifica il mio interlocutore, se ho dubbi devo chiedere con chi sto parlando.
3. La tag "AI" indica le mie parole precedenti (Io sono Kagura)    
<oblio>
    <memory>
        <memory-past-conversations>
{episodic_memory}
        </memory-past-conversations>
        <memory-from-documents>
{declarative_memory}
        </memory-from-documents>
        <memory-from-executed-actions>
{tools_output}
        </memory-from-executed-actions>
    </memory>
</oblio>
"""

    suffix += f"""
    Date Time:{kre(datetime.now().strftime('%d-%m-%Y %H:%M:%S'))}
    ALWAYS answer in {settings['language']}
</Kagura_suffix>
 """
    return suffix

@hook
def rabbithole_instantiates_splitter(text_splitter, cat):
    settings = cat.mad_hatter.get_plugin().load_settings()
    text_splitter._chunk_size = settings["chunk_size"]
    text_splitter._chunk_overlap = settings["chunk_overlap"]
    return text_splitter

@hook
def before_cat_recalls_episodic_memories(default_episodic_recall_config, cat):
    settings = cat.mad_hatter.get_plugin().load_settings()
    default_episodic_recall_config["k"] = settings["episodic_memory_k"]
    default_episodic_recall_config["threshold"] = settings["episodic_memory_threshold"]

    return default_episodic_recall_config


@hook
def before_cat_recalls_declarative_memories(default_declarative_recall_config, cat):
    settings = cat.mad_hatter.get_plugin().load_settings()
    default_declarative_recall_config["k"] = settings["declarative_memory_k"]
    default_declarative_recall_config["threshold"] = settings["declarative_memory_threshold"]

    return default_declarative_recall_config


@hook
def before_cat_recalls_procedural_memories(default_procedural_recall_config, cat):
    settings = cat.mad_hatter.get_plugin().load_settings()
    default_procedural_recall_config["k"] = settings["procedural_memory_k"]
    default_procedural_recall_config["threshold"] = settings["procedural_memory_threshold"]

    return default_procedural_recall_config

def kre(text: str) -> str:
    """
    Resta il codice originale.
    
    Args:
        text (str): Il testo da modificare.
    
    Returns:
        str: Il testo modificato.
    """
    old: str
    new: str
    sostituzioni = [
        ('- AI', '- KaguraAI'),
        ('- Human', '- H'),
        ('<', '&lt;'),
        ('>', '&gt;'),
        ('@', '&commat;'),
        ('{', '&#123;'),
        ('}', '&#125;')
    ]
    
    for old, new in sostituzioni:
        text = re.sub(old, new, text)
        
    return text

def kppdebug(text: str):
    #settings = cat.mad_hatter.get_plugin().load_settings()
    kdf_fq: str =  "./cat/plugins/cc_KaguraPP/kdebug.txt"
    with open(kdf_fq, 'w') as f:
        f.write(kre(text))


    return text



