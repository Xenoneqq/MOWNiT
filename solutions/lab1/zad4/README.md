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
\Delta \eta_{S1} = \sqrt{(1.5\%)^2 + (1.0\%)^2 + (3.6\%)^2} = \sqrt{0.000225 + 0.0001 + 0.001296} = \sqrt{0.001621} \approx 3.6\%
```

```math
\eta_{S1} = 0.76 \pm 0.027
```

Zakres możliwych wartości: \( 0.76 - 0.027 = 0.733 \) do \( 0.76 + 0.027 = 0.787 \).

### Obliczenia dla kolektora S2:

```math
\Delta \eta_{S2} = \sqrt{(0.5\%)^2 + (1.0\%)^2 + (2.0\%)^2} = \sqrt{0.0025 + 0.0001 + 0.0004} = \sqrt{0.003} \approx 5.48\%
```

```math
\eta_{S2} = 0.70 \pm 0.038
```

Zakres możliwych wartości: \( 0.70 - 0.038 = 0.662 \) do \( 0.70 + 0.038 = 0.738 \).

### Czy S1 ma większą sprawność niż S2?

Ponieważ zakresy wartości η się nakładają

```
zakres S1 = (0.733 - 0.787)
zakres S2 = (0.662 - 0.738)
```

nie możemy stwierdzić z pewnością, że S1 jest bardziej efektywny niż S2.
