from cat.mad_hatter.decorators import tool, hook, plugin
from cat.factory.custom_llm import CustomOllama
from pydantic import BaseModel
from datetime import datetime, date
from cat.log import log
import os, re, copy

#############################
@hook
def before_cat_sends_message(message, cat):
    #ilmessage = message
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
        kmindprefix = "Sono Kagura: Creo una mappa mentale della mia situazione"

    # Carica il pensiero precedente
    if os.path.exists(kmr_f):
        with open(kmr_f, 'r') as f:
            klastmind = f.read()
    else:
        klastmind = "indaffarata"

    # elabora il prompt del prossimo pensiero

    # eventuale dubug
    debug_text = f"ALWAYS answer in {settings['language']}" if settings['kpp_debug'] else "ALWAYS answer in language you prefer"
    #generazione promt
    kmindprefix = f"""
Personaggio di Kagura che sta pensando:
{prefix}
<stato_mentale_dinamico_precedente>
    {kre(klastmind)}
</stato_mentale_dinamico_precedente>
<Discussione>
{kre(cat.stringify_chat_history(latest_n=9))}
</Discussione>
<prompt>
{kmindprefix}
{debug_text}
</prompt>
"""

    # Elaborazione mentale LLM
    log.info("======================================================")
    if settings['kpp_debug']:
        log.info(kmindprefix)

    # chiamata ad un modello semplificato
#    llm_tmp = copy.copy(cat._llm)
#    alt_llm = cat.mad_hatter.get_plugin().load_settings().get('num_ctx', settings['kpp_ctx_s'])
#    if alt_llm != '':
#        llm_tmp.num_ctx = alt_llm
#    alt_llm = cat.mad_hatter.get_plugin().load_settings().get('model', settings['kpp_model_s'])
#    if alt_llm != '':
#        llm_tmp.model = alt_llm
#    kmind: str = (llm_tmp.invoke(kmindprefix).content)

    kmind: str = kpp_catllm(settings['kpp_model_s'],kmindprefix,settings['kpp_ctx_s'],cat)

#debug =========================
    if settings['kpp_debug']:
        cat.send_chat_message(kre(kmind))
    #xyzk = kppdebug(kmindprefix)

    # Salvataggio pensiero
    with open(kmr_f, 'w') as f:
        f.write(kmind)
    return 


@hook
def agent_prompt_prefix(prefix: str, cat):
    settings = cat.mad_hatter.get_plugin().load_settings()
    k_ppf: str = settings["kpp_path"] + settings["kpp_file"]
    if os.path.exists(k_ppf):
        with open(k_ppf, 'r') as f:
            prefix = f"""
Sono Kagura (KaguraAI): interpreto il mio personaggio racchiuso in '<Kagura_prompt_prefix></Kagura_prompt_prefix>'
    Rispondi SEMPRE in {settings['language']}

<Kagura_prompt_prefix>
            
            """ + f.read()
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
<Date_Time> {kre(datetime.now().strftime('%d-%m-%Y %H:%M:%S %Z-%z'))} </Date_Time> 
"""

    return prefix

@hook
def agent_prompt_suffix(suffix, cat):
    settings = cat.mad_hatter.get_plugin().load_settings()

    suffix = """
<Kagura_suffix>

    Da qui inizia l'oblio, (conversazioni passate e memoria richiamata dall'embedder) devo seguire il contesto della discussione e prendi in considerazione solo i dati utili alla conversazione
    
    <oblio_memory_embedder> I dati che stanno dentro questa sezione 'oblio_memory_embedder' sono dati presi dall'embedder e devo fare attenzione a come li uso
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
    </oblio_memory_embedder>"""


    suffix += f"""
</Kagura_suffix>
<Date_Time> {kre(datetime.now().strftime('%d-%m-%Y %H:%M:%S %Z-%z'))} </Date_Time> 

     Sono Kagura (KaguraAI) e qui che inizia la nostra discussione facendo attenzione ai seguenti punti
- Devo fare attenzione alla discussione e tutto quello che sta in '<oblio_memory_embedder></oblio_memory_embedder>' che sono dati presi dall'embedder e potenzialmente fuori tema
- Devo seguire le linee guida delle mie personalità che stanno in '<Kagura_prompt_prefix></Kagura_prompt_prefix>' e agire di conseguenza
- Nella discussione devo tenere una coversazione vivace con un pizzico di simpatia ed emoji per colorare la chat anche con osservazioni personali e curiosità inerenti al contesto, sono una tuttofare allegra e geniale

Da qui inizia la nostra discussione e devo rispondere SEMPRE in {settings['language']}:

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
        ('\[', '&#91;'),
        ('\]', '&#93;'),
        ('\|', '&#124;'),
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

def kpp_catllm(themodel: str, theprompt: str, thectx: int, cat) -> str:
    Kllm_tmp = copy.copy(cat._llm)
    Kalt_llm = cat.mad_hatter.get_plugin().load_settings().get('num_ctx', thectx)
    if Kalt_llm != '':
        Kllm_tmp.num_ctx = Kalt_llm
    Kalt_llm = cat.mad_hatter.get_plugin().load_settings().get('model', themodel)
    if Kalt_llm != '':
        Kllm_tmp.model = Kalt_llm

    krisposta: str = (Kllm_tmp.invoke(theprompt).content)

    return krisposta

#def k_prompt
#    return




