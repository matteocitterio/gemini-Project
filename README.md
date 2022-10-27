# Gemini-Project
One code to rule them all


# Intresting works
## Datasets site
https://www.aeroclubmilano.it/meteo/about/

## Small meteorological forecast
https://www.microsoft.com/en-us/research/blog/micro-climate-predictions-enabling-hyper-local-decisions-for-agriculture-and-renewables/


In questa repository viene postato tutto l'occorrente per il deploy iniziale da parte di ogni utente del proprio token e pool.

### Contenuto:

- `Bot_Minter_Data.json`: contiene l'indirizzo del bot minter e il suo mnemonic. Contiene inoltre l'indirizzo del contratto `Paycoin` di cui è Minter. WARNING: OGNUNO DOVRà GIRARE 0.2 ETH A QUESTO ACCOUNT PER TUTTI I DEPLOY E I MINTING
- `Starting_Pool_Ratios.json`: contiene la quantità iniziale di token e paycoin che andranno mintati alla pool di ciascuno. WARNING: DA AGGIUNGERE INDIRIZZI DI DIANA RICCARDO E FRANCESCO. DA CONCORDARE I RAPPORTI ESATTI.
- `initial_deploy.py`: script per effettuare il deploy. E' necessario che i due precedenti file si trovino nella stessa cartella di quest'ultimo per il funzionamento.
- `StartingPoolRatios_to_JSON.py`: script python che serve solamente per creare un JSON con i rapporti iniziali di liquidità che saranno presenti nelle pool di ciasscuno di noi.
- `addresses.json`: TEMPLATE del file .json che costruiremo una volta effettutato il deploy contenente tutte le informazioni pubbliche degli utenti e del bot minter.
- `addresses_TO_JSON.py`: prenderà tutti i file `YOURNAME_public_dict.json` che si vengono a creare dopo il deploy di ciascuno e costruisce in automatico il file `addresses.json`
- `paycoin_deploy.py`: esegue il deploy di un contratto `Paycoin` e di uno `Challenge` di cui `BotMinter` sarà proprietario. Distribuisce inoltre a tutti i giocatori 50k PcN.
