# Gemini-Project
One code to rule them all


# Intresting works
## Datasets site
https://www.aeroclubmilano.it/meteo/about/

## Small meteorological forecast
https://www.microsoft.com/en-us/research/blog/micro-climate-predictions-enabling-hyper-local-decisions-for-agriculture-and-renewables/


In questa repository viene postato tutto l'occorrente per il deploy iniziale da parte di ogni utente del proprio token e pool.

### Tings to address:

- é un problema trainare il modello senza stare attenti al fatto che non abbiamo sempre gli stessi dati nello stesso arco temporale data dall'assenza di questi o dal drop dei nan? I mean: da una parte in un caso reale vorremmo che il modello sia in grado di dare predizioni anche con buchi nei dati qua e là, dall'altra cè il rischio che si perda la coerenza temporale. Fare la più stupida delle interpolazioni lineari quando manca un dato?

- How to properly split our dataset into training - val - test. The article reported above splits them in a 60-20-20 manner.

