### **Wnioski z wykresu błędów interpolacji dla \(f_1(x)\) i \( f_2(x) \)**

#### **1. Najmniej dokładna metoda: interpolacja wielomianowa Lagrange’a z równoodległymi węzłami**

- Błąd rośnie wykładniczo wraz ze wzrostem liczby węzłów \( n \).
- Występuje **efekt Rungego** – dla dużych \( n \) wielomian silnie oscyluje na końcach przedziału, co prowadzi do ogromnych błędów (ponad \( 10^{26} \)!).
- Metoda jest **niestabilna** dla dużych \( n \) i nie nadaje się do precyzyjnej interpolacji w tej postaci.

#### **2. Średnio dokładna metoda: interpolacja wielomianowa Lagrange’a z węzłami Czebyszewa**

- Działa lepiej niż węzły równoodległe – efekt Rungego jest mniejszy, ale nadal obecny.
- Błąd rośnie znacznie wolniej niż w przypadku węzłów równoodległych, ale dla dużych \( n \) także zaczyna gwałtownie rosnąć.
- Lepszy wybór niż węzły równoodległe, ale nadal nie jest to metoda idealna.

#### **3. Najbardziej dokładna metoda: interpolacja funkcją sklejaną (spline cubic)**

- Zachowuje stabilność i niski błąd nawet dla dużych \( n \).
- Brak efektu Rungego – błąd pozostaje niewielki, co świadczy o dobrze dopasowanej aproksymacji funkcji.
- **Najlepszy wybór do interpolacji**, ponieważ nie tylko zapewnia dokładność, ale również unika problemów z oscylacjami.

### **Podsumowanie:**

**Najbardziej dokładna metoda** to interpolacja **funkcją sklejaną (spline cubic)**, ponieważ jej błąd jest najmniejszy i nie rośnie dla dużych \( n \).  
**Najmniej dokładna metoda** to interpolacja wielomianowa Lagrange’a **z równoodległymi węzłami**, ponieważ powoduje ogromne błędy i efekt Rungego.
