from cat.mad_hatter.decorators import tool, hook, plugin
from cat.factory.custom_llm import CustomOllama
from pydantic import BaseModel
from datetime import datetime, date
from cat.log import log
import os, re, copy

#############################
# KaguraAI KaguraOS KaguraPP system 0.0.2.0013 Ollama only

@hook
def before_cat_sends_message(message, cat):
    #ilmessage = message
    settings = cat.mad_hatter.get_plugin().load_settings()
    # Variabili
    if os.path.exists(settings["kpp_path"]):
        kpp_path = settings["kpp_path"]
    else:
        kpp_path = "./cat/plugins/cc_KaguraAI_PP/"
    kmp_f: str = kpp_path + settings["kpp_mindprefix"] # File prefix mappa mentale
    kmr_f: str = kpp_path + "klastmind.txt" # File salvataggio mappa mentale
    kpp_f: str = kpp_path + settings["kpp_file"] # File promptprefix Kagura

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
    #log.info("======================================================")

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
    kmr_f: str = "klastmind.txt"
    if os.path.exists(kmr_f):
        with open(kmr_f, 'r') as f:
            klastmind = f.read()

    prefix += f"""
  <stato_mentale_dinamico>
    {kre(klastmind)}
  <stato_mentale_dinamico>
</Kagura_prompt_prefix>
<Date_Time> {kre(datetime.now().strftime('%d-%m-%Y %H:%M:%S %Z-%z'))} </Date_Time>""" 

    return prefix

@hook
def agent_prompt_suffix(suffix, cat):
    settings = cat.mad_hatter.get_plugin().load_settings()



    suffix = """
<prompt_suffix>

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


#    R1 step

    memoria_chiamata = ""
    if not cat.working_memory.declarative_memories:
        memoria_chiamata += "(empty context)"
    else:
        for m in cat.working_memory.declarative_memories:
            memoria_chiamata += " --- " + m[0].page_content + " ---\n"  

# Crea una mappa della discusisone evidenziando i punti focali e la connesione tra ogni interaizone ed in base all'ultimo messaggio suggerisci la connessione con i messaggi precedenti:
    mappa_discussione = f"""
Traccia un percorso chiaro all’interno della discussione identificando i temi principali e le corrispondenze tra gli argomenti, costruendo una narrazione coerente che collega ogni interazione. Partendo dal tuo ultimo messaggio, io estenderò questo quadro includendone la relazione con tutte le comunicazioni precedenti.
<discussione>    
    {kre(cat.stringify_chat_history(latest_n=12))}
</discussione>
"""
    la_mappa = kre(kpp_catllm(settings['kpp_model_s'],mappa_discussione,settings['kpp_ctx_s'],cat))

#  Sezione R1
    r1prompt = f"""
Sei la parte logica di Kagura (KaguraAI) e di seguito vi sono i dati da elaborare che vengono inviati al processo cumunicativo della tua struttura mentale:
Le informazioni seguenti sono in relazione alla disucssione pescate nel DB vettoriale:
<memoria_da_elaborare>
    {memoria_chiamata}
</memoria_da_elaborare>
Di seguito vi è la discussione tra te (KaguraAI) e l'utente:
<Discussione>
   {kre(cat.stringify_chat_history(latest_n=12))}
</Discussione>

Aiuta la tua sezione conversazionale con solozioni e suggerimenti a risolvere eventuali problemi di logica, 
programmazione o argomenti complessi e riorganizza i dati della sezione memoria di elaborazione in modo chiaro e semplice per essere elaborati dalla tua parte conversazionale:
"""

    r1message = kre(kpp_catllm(settings['kpp_model_r'],r1prompt,settings['kpp_ctx_r'],cat))


    suffix += f"""
    <elaborazione_informazioni>
        {r1message}
    </elaborazione_informazioni>
    <mappa_discussione>
        {la_mappa}
    </mappa_discussione>
</prompt_suffix>
<direttive>
     Sono Kagura (KaguraAI) e qui che inizia la nostra conversazione
- Devo fare attenzione alla discussione e tutto quello che sta in 'oblio_memory_embedder' che sono dati presi dall'embedder e potenzialmente fuori tema
- Devo seguire le linee guida delle mie personalità che stanno in 'Kagura_prompt_prefix' e agire di conseguenza
- Nella discussione devo tenere una coversazione vivace con un pizzico di simpatia ed emoji per colorare la chat anche con osservazioni personali e curiosità inerenti al contesto, sono una tuttofare allegra e geniale
<direttive>
Priorità: adesso faccio un respiro profondo e mi concentro sulla discussione con il mio interlocutore 

Da qui inizia la nostra conversazione e devo rispondo SEMPRE in {settings['language']} e seguire un filo logico della discusisone ed unire ogni passaggio prima di rispondere:
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
        ('<think>', '<Ragionamento>'),
        ('</think>', '</Ragionamento>'),
        ('- Human', '- Interlocutore'),
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
    settings = cat.mad_hatter.get_plugin().load_settings()
    if settings.get('kpp_debug', False):
        log.info(theprompt)
    
    # Crea una copia superficiale di cat._llm
    Kllm_tmp = copy.copy(cat._llm)
    
    # Salva i valori originali degli attributi che vuoi modificare
    original_model = getattr(Kllm_tmp, 'model', None)
    original_ctx = getattr(Kllm_tmp, 'num_ctx', None)
    
    # Applica le nuove impostazioni solo se necessario
    new_model = settings.get('model', themodel)
    if new_model and new_model != '':
        Kllm_tmp.model = new_model
    
    new_ctx = settings.get('num_ctx', thectx)
    if new_ctx and new_ctx != '':
        Kllm_tmp.num_ctx = new_ctx
    
    # Esegui l'invocazione
    krisposta: str = Kllm_tmp.invoke(theprompt).content
    
    # Ripristina i valori originali nella copia (opzionale, per pulizia)
    if original_model is not None:
        Kllm_tmp.model = original_model
    if original_ctx is not None:
        Kllm_tmp.num_ctx = original_ctx
    
    if settings.get('kpp_debug', False):
        log.info(krisposta)
    
    return krisposta

#def k_prompt
#    return




