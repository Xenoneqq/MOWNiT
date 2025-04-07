### Czemu najniższy błąd względny jest przy \( m = 6 \)?

Wielomian 6. stopnia ma aż 7 parametrów (bo \( k = m + 1 = 7 \)), więc może **idealnie dopasować się** do 9 punktów danych.

To prowadzi do **bardzo niskiego błędu na znanych danych** – ale nie oznacza, że dobrze poradzi sobie z nowymi (czyli nieznanymi) danymi, np. prognozowaniem populacji w przyszłości.

Zjawisko to nazywamy **przeuczeniem (overfittingiem)** – model nauczył się szumu zamiast trendu.

---

### Czemu najlepszy AICc jest przy \( m = 2 \)?

AICc nie patrzy tylko na dopasowanie, ale także **karze za złożoność modelu**.

Dzięki temu wybiera taki stopień wielomianu, który **dobrze dopasowuje dane, ale nie jest zbyt skomplikowany**.

Przy małej próbce (jak tutaj: tylko 9 punktów), AICc **szczególnie mocno penalizuje nadmiar parametrów**.
