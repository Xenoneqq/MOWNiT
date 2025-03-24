## Zadanie 4: Analiza niepewności sprawności kolektorów

Efektywność \( \eta \) kolektora słonecznego dana jest wzorem:

```math
\eta = K \frac{Q T_d}{I}
```

Zmienna \( K \) jest stałą, więc jej błąd nie wpływa na niepewność względną \( \eta \). Błąd względny \( \eta \) obliczamy jako:

```math
\Delta \eta = |\Delta Q| + |\Delta T_d| + |\Delta I|
```

### Obliczenia dla kolektora S1:

```math
\Delta \eta_{S1} = |1.5\%| + |1.0\%| + |3.6\%| = 6.1\%
```

```math
\eta_{S1} = 0.76 \pm (0.76 \times 0.061) = 0.76 \pm 0.046
```

Zakres możliwych wartości:

```math
0.76 - 0.046 = 0.714
```
```math
0.76 + 0.046 = 0.806
```

### Obliczenia dla kolektora S2:

```math
\Delta \eta_{S2} = |0.5\%| + |1.0\%| + |2.0\%| = 3.5\%
```

```math
\eta_{S2} = 0.70 \pm (0.70 \times 0.035) = 0.70 \pm 0.025
```

Zakres możliwych wartości:

```math
0.70 - 0.025 = 0.675
```
```math
0.70 + 0.025 = 0.725
```

### Czy S1 ma większą sprawność niż S2?

Zakresy sprawności to:

```math
S1: (0.714 - 0.806)
```
```math
S2: (0.675 - 0.725)
```

Zakresy **nadal się nakładają** (0.714 – 0.725), więc **wciąż nie możemy jednoznacznie stwierdzić, że S1 jest bardziej efektywny niż S2**.

