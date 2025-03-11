## Zadanie 4: Analiza niepewności sprawności kolektorów

Efektywność η kolektora słonecznego dana jest wzorem:
```math
\eta = K \frac{Q T_d}{I}
```
Zmienna K jest stałą, więc jej błąd nie wpływa na niepewność względną η. Błąd względny η obliczamy jako:
```math
\Delta \eta = \sqrt{(\Delta Q)^2 + (\Delta T_d)^2 + (\Delta I)^2}
```

### Obliczenia dla kolektora S1:
```math
\Delta \eta_{S1} = \sqrt{(1.5\%)^2 + (1.0\%)^2 + (3.6\%)^2} = \sqrt{0.0225 + 0.01 + 0.1296} = \sqrt{0.1621} \approx 12.7\%
```
```math
\eta_{S1} = 0.76 \pm 0.127
```
Zakres możliwych wartości: \( 0.76 - 0.127 = 0.633 \) do \( 0.76 + 0.127 = 0.887 \).

### Obliczenia dla kolektora S2:
```math
\Delta \eta_{S2} = \sqrt{(0.5\%)^2 + (1.0\%)^2 + (2.0\%)^2} = \sqrt{0.0025 + 0.01 + 0.04} = \sqrt{0.0525} \approx 7.25\%
```
```math
\eta_{S2} = 0.70 \pm 0.051
```
Zakres możliwych wartości: \( 0.70 - 0.051 = 0.649 \) do \( 0.70 + 0.051 = 0.751 \).

### Czy S1 ma większą sprawność niż S2?
Ponieważ zakresy wartości η się nakładają 
```
zakres S1 = (0.649 - 0.751)
zakres S2 = (0.633 - 0.887)
```
nie możemy stwierdzić z pewnością, że S1 jest bardziej efektywny niż S2.

