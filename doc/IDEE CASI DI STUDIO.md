# IDEE CASI DI STUDIO

## Evasione della Moderazione dei Contenuti (Tecnica: Evade AI Model / LLM Prompt Obfuscation)

ABSTRACT:

 I sistemi di moderazione dei contenuti basati su AI sono costantemente sfidati da utenti che sviluppano metodi creativi per eludere il rilevamento. Questo caso di studio esamina l'uso dell' "algospeak" — un linguaggio in codice in continua evoluzione — come principale tecnica di "Defense Evasion" secondo il framework MITRE ATLAS. Analizzando esempi concreti tratti dalle principali piattaforme social, la ricerca identifica le metodologie chiave di offuscamento utilizzate per veicolare discorsi d'odio, disinformazione e altri contenuti vietati. Il progetto evidenzia i limiti intrinseci della moderazione automatica di fronte alla creatività linguistica umana e alla rapida evoluzione delle norme culturali online, sottolineando l'urgenza di sviluppare contromisure più resilienti e contestuali.


## LLM jailbreaking

ABSTRACT:

L'avvento dei Large Language Models (LLM) ha introdotto anche nuove e complesse superfici d'attacco. Per garantire un comportamento sicuro ed etico, questi modelli sono dotati di robusti meccanismi di sicurezza, noti come "alignment". Tuttavia, questi meccanismi sono vulnerabili a tecniche di "jailbreaking", che consentono a un utente malintenzionato di bypassare le restrizioni imposte. Questo caso di studio analizza questa minaccia critica, inquadrandola nella tattica di "Defense Evasion" della matrice MITRE ATLAS. Attraverso l'analisi dettagliata della tecnica del "LLM Prompt Injection", verranno esaminati casi di studio emblematici, come l'attacco "DAN" (Do Anything Now) e le sue varianti, che sfruttano il role-playing e la manipolazione del contesto. Lo studio cataloga le diverse strategie di attacco, dimostrando come la manipolazione del linguaggio naturale possa indurre i modelli a violare le proprie direttive etiche per generare disinformazione, codice malevolo o altri contenuti dannosi.
Durante una fase cruciale del suo addestramento, chiamata Reinforcement Learning from Human Feedback (RLHF), al modello vengono mostrati esempi di risposte buone e cattive. Viene premiato quando si rifiuta di generare contenuti dannosi e "punito" quando lo fa. Questo processo costruisce delle "barriere etiche" interne, delle vere e proprie direttive che gli dicono: "Non creare discorsi d'odio", "Non fornire istruzioni per attività illegali", "Non generare codice malevolo".
Qui entra in gioco il "jailbreaking". L'attaccante non cerca una vulnerabilità nel codice del server su cui gira l'IA, ma una falla nella sua logica. L'obiettivo è convincere il modello che, in una specifica situazione, le sue regole non si applicano. Ecco perché si classifica come "Defense Evasion"

## Phishing: (ATTACK & DEFENSE)

ABSTRACT:

I gateway di posta elettronica sono diventati molto bravi a scansionare il testo delle email per individuare URL malevoli. Per aggirare queste difese, gli attaccanti stanno tornando a usare elementi visivi: principalmente i QR code. Questo fenomeno è noto come "Quishing" (QR Code Phishing).

Il Ruolo della Computer Vision (in Attacco e Difesa):
- Attacco: L'intero attacco si basa sul fatto che l'informazione è in un formato leggibile dalle macchine (CV del telefono) ma opaco per le difese testuali.

- Difesa: Una possibile contromisura avanzata consiste nell'utilizzare la Computer Vision. Un sistema di sicurezza potrebbe:
    - Riconoscere la presenza di QR code nelle email.  
    - "Leggerli" in un ambiente sicuro (sandbox).
    - Analizzare l'URL di destinazione per determinare se è malevolo.

Come Funziona l'Attacco:
- L'Email: Un attaccante invia un'email che sembra legittima (es. una notifica per l'autenticazione a due fattori, un avviso di pacco in consegna). Invece di un link testuale, l'email contiene solo un'immagine di un QR code.
- L'Evasione: Le difese automatiche che scansionano il testo non vedono URL sospetti, perché l'URL è "codificato" nell'immagine del QR code. L'email supera i filtri.
- L'Utente: L'utente, abituato a usare i QR code per qualsiasi cosa (menu, pagamenti), inquadra il codice con il proprio smartphone.
La Compromissione: Il QR code lo reindirizza a una pagina di phishing identica a quella legittima, inducendolo a inserire le proprie credenziali. Il telefono, spesso meno protetto di un PC aziendale, diventa il punto di ingresso.

## Furto d'Identità Digitale: Deepfake (L'Attacco Biometrico)

ABSTRACT:


Usiamo il nostro volto per sbloccare telefoni, approvare pagamenti e verificare la nostra identità online (processi KYC - Know Your Customer). Questi sistemi si basano su modelli di CV per riconoscere un volto e, soprattutto, per assicurarsi che sia una persona reale e viva.

 Un attaccante raccoglie foto e video della vittima dai social media. Usa queste informazioni per addestrare un modello generativo (GAN o simile) e creare un "deepfake": un video sintetico del volto della vittima che può essere manipolato in tempo reale. L'attaccante punta la fotocamera del suo PC verso questo video deepfake e, quando il sistema di sicurezza chiede di "sorridere", lui fa sorridere il suo avatar digitale, bypassando il controllo di vivacità.

 ## Attacchi Avversari a Sistemi di Visione Artificiale nel Mondo Reale (esempio segnali stradali) DEEFENSE EVASION - AI MODEL

 ABSTRACT:

Un attaccante non agisce a livello digitale, ma crea oggetti fisici progettati per ingannare un modello di CV.  Ricercatori hanno dimostrato come apporre degli adesivi neri e bianchi, apparentemente casuali, su un segnale di "STOP" possa indurre il sistema di visione di un'auto a classificarlo erroneamente come un segnale di "Limite di Velocità". L'occhio umano non nota quasi l'inganno, ma per la macchina la percezione è completamente alterata. 

Oppure, Un attaccante indossa una felpa con una stampa speciale ("adversarial patch"). Questa stampa è stata generata da un algoritmo per "confondere" i modelli di rilevamento di persone. Chi la indossa diventa quasi "invisibile" alle telecamere di sorveglianza, che non riescono più a disegnare un riquadro attorno alla sua figura.
