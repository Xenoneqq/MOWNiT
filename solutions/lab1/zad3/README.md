## Zadanie 3: Unikanie zjawiska kancelacji

Unikanie kancelacji ma na celu usunięcie ryzyka utraty danych wynikających z zaokrągleń danych operacji matematycznych np. odejmowania które jest operacją podatną na utratę danych.

### Podpunkt A

Aby uniknąć zjawiska kancelacji, możemy pomnożyć przez sprzężenie:
```math
\sqrt{x+1} - 1 = \frac{(\sqrt{x+1} - 1)(\sqrt{x+1} + 1)}{\sqrt{x+1} + 1} = \frac{x}{\sqrt{x+1} + 1}
```
W ten sposób otrzymujemy równanie, które zamiast korzystania z operacji odejmowania używa dzielenia, które jest stabliniejsze.
### Podpunkt B

Rozkładamy na iloczyn:
```math
x^2 - y^2 = (x - y)(x + y)
```
Zamiast odejmowania dużych liczb, operujemy na ich różnicy. W ten sposób dla bardzo bliskich sobie liczb x ~ y wykorzystujemy iloczyn z pojedyńczą operacją odejmowania.

### Podpunkt C

Mnożymy przez sprzężenie:
```math
1 - \cos x = \frac{(1 - \cos x)(1 + \cos x)}{1 + \cos x} = \frac{\sin^2 x}{1 + \cos x}
```
Dzięki temu unikamy bezpośredniego odejmowania wartości bliskich sobie, co mogłoby prowadzić do utraty precyzji.

### Podpunkt D

Korzystamy z tożsamości trygonometrycznej:
```math
\cos^2 x - \sin^2 x = \cos 2x
```
Użycie pojedyńczej funkcji trygonometrycznej jest dokładniejsze niż korzystając z różnicy cos^2(x) - sin^2(x).

### Podpunkt E

Przekształcamy:
```math
\ln x - 1 = \ln \frac{x}{e}
```
Dla podpunktu E, możemy przybliżyć wartość lx(x) - 1 zamieniając 1 w ln(e) i sprowadzając do pojedyńczego logarytmu. W ten sposób unikamy odejmowania które jest dużo mniej stabline niż dzielenie.

### Podpunkt F

Rozwijamy w szereg Taylora:
```math
e^x - e^{-x} = 2x + \frac{2x^3}{3!} + \mathcal{O}(x^5)
```
Dla małych \( x \) najlepszym przybliżeniem jest:
```math
e^x - e^{-x} \approx 2x
```
To rozwinięcie pozwala uniknąć odejmowania niemal równych liczb  i , co mogłoby prowadzić do znacznej utraty precyzji w pobliżu zera.
