"""
SRBIN Nikola Tesla, za sva vremena, najveci naucnik sveta.

SERBIAN Nikola Tesla, for all time, the greatest scientist in the world.
"""



"""
Tesla_Scalar_1.py  —  SLW motor (cista fizika)

Simulira skalarno-longitudinalni talas (SLW):
- skalarno polje S(x, t) koje se prostire u +x pravcu (d'Alamberova jednacina),
- uzduzno polje E_x = -dS/dx  (gradient-driven, u pravcu prostiranja).

Teorija (Tesla scalar / longitudinalni talasi)
Ima recenziran rad o Extended Electrodynamics (EED)
koji formalno predviđa scalar-longitudinal wave (SLW)
— talas sa električnim poljem u pravcu prostiranja (MDPI Symmetry, 2020).
Čak postoji i američki patent (9,306,527) za antene za takve talase.
Dakle, nije „samo pseudonauka" — ima ozbiljnih pokušaja formalizacije.
"""


from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 39

# Tezine za kombinovani skor (talas vs prava frekvencija).
W_TALAS = 0.7
W_FREQ = 0.3

# Korak 2: primena talasa na CSV.
CSV_PATH = Path("/data/loto7hh_4632_k47.csv")
MIN_BROJ = 1
MAX_BROJ = 39
KOLONE = [f"Num{i}" for i in range(1, 8)]
OUTPUT_DIR = Path(__file__).resolve().parent


def simuliraj_slw(
    duzina=200.0,     # duzina domena (proizvoljne jedinice)
    nx=None,          # broj tacaka po prostoru (= broj kombinacija u CSV)
    c=1.0,            # brzina prostiranja talasa
    cfl=0.5,          # Courant broj (stabilnost: <= 1)
    koraka=600,       # broj vremenskih koraka
    centar=40.0,      # pocetni polozaj pulsa
    sirina=6.0,       # sirina pocetnog Gaussovog pulsa
):
    """Vrati (x, S, E_x): polozaj, skalarno polje i uzduzno polje na kraju simulacije."""
    dx = duzina / (nx - 1)
    dt = cfl * dx / c
    x = np.linspace(0.0, duzina, nx)

    # Pocetni Gaussov puls koji se krece u +x (zadajem S i njegov pomak unazad).
    def gauss(xx):
        return np.exp(-((xx - centar) ** 2) / (2.0 * sirina ** 2))

    S_prev = gauss(x)
    S_curr = gauss(x - c * dt)

    r2 = (c * dt / dx) ** 2
    for _ in range(koraka):
        S_next = np.empty_like(S_curr)
        S_next[1:-1] = (
            2.0 * S_curr[1:-1]
            - S_prev[1:-1]
            + r2 * (S_curr[2:] - 2.0 * S_curr[1:-1] + S_curr[:-2])
        )

        # Otvorene granice: prost prenos susedne vrednosti da puls ne eksplodira na ivici.
        S_next[0] = S_next[1]
        S_next[-1] = S_next[-2]

        S_prev, S_curr = S_curr, S_next

    E_x = -np.gradient(S_curr, dx)
    return x, S_curr, E_x


def glavne_mere(S, E_x):
    """Vrati osnovne mere polja: maksimum S, maksimum |E_x| i energijsku gustinu."""
    gustina_energije = 0.5 * (S ** 2 + E_x ** 2)
    return {
        "max_S": float(np.max(S)),
        "max_abs_E_x": float(np.max(np.abs(E_x))),
        "ukupna_gustina_energije": float(np.sum(gustina_energije)),
    }


# ---------------------------------------------------------------------------
# Korak 2: primena SLW talasa na loto CSV (ne-frekvencijski)
# ---------------------------------------------------------------------------
def ucitaj_izvlacenja(csv_path=CSV_PATH):
    """Ucitaj CSV i vrati listu izvlacenja (svako je lista od 7 brojeva), hronoloski."""
    df = pd.read_csv(csv_path)
    df = df[KOLONE].astype(int)
    return df.to_numpy().tolist()


def ne_frekvencijski_skor(izvlacenja, energija):
    """Za svaki broj 1..39 vrati PROSECNU energiju talasa na mestima gde se pojavio.

    Nije frekvencija (koliko puta), nego gde u talasu broj "lezi" (rezonanca).
    """
    zbir = {b: 0.0 for b in range(MIN_BROJ, MAX_BROJ + 1)}
    pojave = {b: 0 for b in range(MIN_BROJ, MAX_BROJ + 1)}
    for i, red in enumerate(izvlacenja):
        for b in red:
            zbir[b] += energija[i]
            pojave[b] += 1
    skor = {b: (zbir[b] / pojave[b] if pojave[b] > 0 else 0.0) for b in zbir}
    return skor, pojave


def frekvencija_brojeva(izvlacenja):
    """Prava frekvencija: koliko se svaki broj 1..39 pojavio nad svim izvlacenjima.

    Vraca (udeo, pojave): udeo = relativna frekvencija (zbir = 1), pojave = sirovi broj.
    """
    pojave = {b: 0 for b in range(MIN_BROJ, MAX_BROJ + 1)}
    for red in izvlacenja:
        for b in red:
            pojave[b] += 1
    ukupno = sum(pojave.values())
    udeo = {b: (pojave[b] / ukupno if ukupno > 0 else 0.0) for b in pojave}
    return udeo, pojave


def _na_0_1(vrednosti):
    """Min-max skaliranje recnika na opseg 0..1 (ravno na 0 ako su sve jednake)."""
    v = np.array(list(vrednosti.values()), dtype=float)
    raspon = v.max() - v.min()
    if raspon <= 0:
        return {k: 0.0 for k in vrednosti}
    return {k: (vrednosti[k] - v.min()) / raspon for k in vrednosti}


def kombinovani_skor(talas_skor, udeo, w_talas=W_TALAS, w_freq=W_FREQ):
    """Tezinska kombinacija: normalizuj obe komponente na 0..1, pa ponderisan zbir."""
    t = _na_0_1(talas_skor)
    f = _na_0_1(udeo)
    return {b: w_talas * t[b] + w_freq * f[b] for b in talas_skor}


def izaberi_kombinacije(skor, broj_kombinacija=10, seed=SEED):
    """Izaberi 7/39 kombinacije, ponderisano ne-frekvencijskim skorom."""
    rng = np.random.default_rng(seed)
    brojevi = np.array(list(skor.keys()))
    tezine = np.array([skor[b] for b in brojevi], dtype=float)
    if tezine.sum() <= 0:
        tezine = np.ones_like(tezine)
    p = tezine / tezine.sum()

    kombinacije, vidjeno, pokusaji = [], set(), 0
    while len(kombinacije) < broj_kombinacija and pokusaji < broj_kombinacija * 300:
        pokusaji += 1
        izbor = tuple(sorted(rng.choice(brojevi, size=7, replace=False, p=p).tolist()))
        if izbor not in vidjeno:
            vidjeno.add(izbor)
            kombinacije.append(izbor)
    return kombinacije


def skor_kombinacije(kombinacija, skor):
    """Skor kombinacije je zbir skorova njenih 7 brojeva."""
    return float(sum(skor[b] for b in kombinacija))


def nacrtaj_polje(x, S, E_x, osnova="tesla_scalar_1"):
    """Nacrtaj SLW talas (S i E_x) i snimi kao PNG i JPG. Vrati (png, jpg) putanje."""
    energija = 0.5 * (S ** 2 + E_x ** 2)
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("Tesla Scalar - SLW talas (skalarno polje S i uzduzno E_x)")

    ax[0].plot(x, S, color="#1f77b4", lw=0.8)
    ax[0].set_ylabel("S(x)")
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(x, E_x, color="#d62728", lw=0.8)
    ax[1].set_ylabel("E_x = -dS/dx")
    ax[1].grid(True, alpha=0.3)

    ax[2].plot(x, energija, color="#2ca02c", lw=0.8)
    ax[2].set_ylabel("energija")
    ax[2].set_xlabel("x (pozicija)")
    ax[2].grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png = OUTPUT_DIR / f"{osnova}.png"
    jpg = OUTPUT_DIR / f"{osnova}.jpg"
    fig.savefig(png, dpi=150)
    fig.savefig(jpg, dpi=150)
    plt.close(fig)
    print(f"Slika talasa: {png}")
    print(f"Slika talasa: {jpg}")
    return png, jpg


def main():
    # --- Korak 1: SLW motor ---
    izvlacenja = ucitaj_izvlacenja()
    n = len(izvlacenja)
    x, S, E_x = simuliraj_slw(nx=n)
    mere = glavne_mere(S, E_x)
    print()
    print("Tesla Scalar / SLW motor - korak 1")
    print("Talas: skalarno polje S(x,t), prostiranje u +x pravcu")
    print("Uzduzno polje: E_x = -dS/dx")
    print()
    print(f"broj tacaka: {len(x)}")
    print(f"max S: {mere['max_S']:.10f}")
    print(f"max |E_x|: {mere['max_abs_E_x']:.10f}")
    print(f"ukupna gustina energije: {mere['ukupna_gustina_energije']:.10f}")
    print()

    # --- Korak 2: primena talasa na CSV + prava frekvencija ---
    energija = 0.5 * (S ** 2 + E_x ** 2)
    talas_skor, _ = ne_frekvencijski_skor(izvlacenja, energija)
    udeo, pojave = frekvencija_brojeva(izvlacenja)
    skor = kombinovani_skor(talas_skor, udeo)
    poredak = sorted(skor.items(), key=lambda kv: kv[1], reverse=True)
    # Opadajuce po pravoj frekvenciji, pa po velicini broja.
    freq_poredak = sorted(pojave, key=lambda b: (pojave[b], b), reverse=True)
    kombinacije = izaberi_kombinacije(skor, broj_kombinacija=10, seed=SEED)
    rangirane_kombinacije = sorted(
        ((k, skor_kombinacije(k, skor)) for k in kombinacije),
        key=lambda kv: kv[1],
        reverse=True,
    )
    png, jpg = nacrtaj_polje(x, S, E_x, osnova="tesla_scalar_1")

    with open(OUTPUT_DIR / "tesla_scalar_1.txt", "w", encoding="utf-8") as f:
        f.write("Tesla Scalar - korak 2 (tezinski: talas + prava frekvencija)\n")
        f.write(f"CSV: {CSV_PATH}\n")
        f.write(f"Izvlacenja: {n} | Seed: {SEED} | tezine: talas={W_TALAS} freq={W_FREQ}\n\n")
        f.write("Brojevi po kombinovanom skoru (tezinski talas + frekvencija):\n")
        for b, s in poredak:
            f.write(f"  {b:02d}  skor={s:.10f}  freq={udeo[b]:.5f}  (pojava={pojave[b]})\n")

        f.write("\nTabela pravih frekvencija (opadajuce po freq, pa po broju):\n")
        f.write("  broj | pojava |   udeo\n")
        f.write("  -----+--------+--------\n")
        for b in freq_poredak:
            f.write(f"   {b:02d}  |  {pojave[b]:4d}  | {udeo[b]:.5f}\n")
        f.write(f"  ukupno pojava: {sum(pojave.values())}\n")

        f.write("\nPredlozene kombinacije (rangirane po skoru kombinacije):\n")
        for i, (k, s_komb) in enumerate(rangirane_kombinacije, start=1):
            f.write(f"  {i:02d}. " + " ".join(f"{v:02d}" for v in k) + f"  skor_komb={s_komb:.10f}\n")

        f.write("\nSlike talasa/polja:\n")
        f.write(f"  PNG: {png}\n")
        f.write(f"  JPG: {jpg}\n")

    print()
    print("\nTesla Scalar - korak 2 (tezinski: talas + prava frekvencija)")
    print(f"CSV: {CSV_PATH} | Izvlacenja: {n} | tezine: talas={W_TALAS} freq={W_FREQ}")
    print("\nTop 10 brojeva po kombinovanom skoru (tezinski talas + frekvencija):")
    for b, s in poredak[:10]:
        print(f"  {b:02d}  skor={s:.10f}  freq={udeo[b]:.5f}  (pojava={pojave[b]})")
    
    print()
    print("\nTabela pravih frekvencija (opadajuce po freq, pa po broju):")
    print("  broj | pojava |   udeo")
    print("  -----+--------+--------")
    for b in freq_poredak:
        print(f"   {b:02d}  |  {pojave[b]:4d}  | {udeo[b]:.5f}")
    print(f"  ukupno pojava: {sum(pojave.values())}")
    
    print()
    print("\nPredlozene kombinacije (rangirane po skoru kombinacije):")
    for i, (k, s_komb) in enumerate(rangirane_kombinacije, start=1):
        print(f"  {i:02d}. " + " ".join(f"{v:02d}" for v in k) + f"  skor_komb={s_komb:.10f}")
    print(f"\nSacuvano: {OUTPUT_DIR / 'tesla_scalar_1.txt'}")
    print()


if __name__ == "__main__":
    main()



"""
Tesla Scalar / SLW motor - korak 1
Talas: skalarno polje S(x,t), prostiranje u +x pravcu
Uzduzno polje: E_x = -dS/dx

broj tacaka: 4632
max S: 0.9999984566
max |E_x|: 0.1010872708
ukupna gustina energije: 124.8336223276

Slika talasa: /Tesla/tesla_scalar_1.png
Slika talasa: /Tesla/tesla_scalar_1.jpg


Tesla Scalar - korak 2 (tezinski: talas + prava frekvencija)
CSV: /data/loto7hh_4632_k47.csv | Izvlacenja: 4632 | tezine: talas=0.7 freq=0.3

Top 10 brojeva po kombinovanom skoru (tezinski talas + frekvencija):
  34  skor=0.9213793103  freq=0.02692  (pojava=873)
   x  skor=0.8176249656  freq=0.02600  (pojava=843)
  08  skor=0.7603045533  freq=0.02810  (pojava=911)
   y  skor=0.6931601524  freq=0.02551  (pojava=827)
  13  skor=0.6706469920  freq=0.02554  (pojava=828)
   z  skor=0.6127524074  freq=0.02560  (pojava=830)
  33  skor=0.6112555867  freq=0.02634  (pojava=854)
  37  skor=0.5766162438  freq=0.02652  (pojava=860)
  03  skor=0.5593769679  freq=0.02547  (pojava=826)
  25  skor=0.5585228387  freq=0.02591  (pojava=840)


Tabela pravih frekvencija (opadajuce po freq, pa po broju):
  broj | pojava |   udeo
  -----+--------+--------
   08  |   911  | 0.02810
    x  |   905  | 0.02791
   34  |   873  | 0.02692
    y  |   869  | 0.02680
   11  |   861  | 0.02655
    z  |   860  | 0.02652
   32  |   857  | 0.02643
    x  |   854  | 0.02634
   22  |   851  | 0.02625
    y  |   849  | 0.02618
   29  |   849  | 0.02618
    z  |   845  | 0.02606
   07  |   844  | 0.02603
    x  |   843  | 0.02600
   09  |   843  | 0.02600
    y  |   842  | 0.02597
   25  |   840  | 0.02591
    z  |   840  | 0.02591
   16  |   837  | 0.02581
   31  |   830  | 0.02560
   13  |   828  | 0.02554
   05  |   828  | 0.02554
   21  |   827  | 0.02551
   03  |   826  | 0.02547
   02  |   825  | 0.02544
   28  |   821  | 0.02532
   18  |   821  | 0.02532
   06  |   816  | 0.02517
   19  |   814  | 0.02510
   04  |   812  | 0.02504
   12  |   810  | 0.02498
   14  |   809  | 0.02495
   15  |   798  | 0.02461
   27  |   789  | 0.02433
   01  |   788  | 0.02430
   30  |   787  | 0.02427
   36  |   786  | 0.02424
   20  |   770  | 0.02375
   17  |   766  | 0.02362
  ukupno pojava: 32424


Predlozene kombinacije (rangirane po skoru kombinacije):
  01. x 12 13 y 34 z 39  skor_komb=4.5418643612
  02. x 08 13 y 24 z 34  skor_komb=4.1634211362
  03. x 17 18 y 26 z 34  skor_komb=3.9810045066
  04. x 11 16 y 25 z 39  skor_komb=3.9122219361
  05. x 25 28 y 31 z 34  skor_komb=3.7585875317
  06. x 07 13 y 32 z 39  skor_komb=3.6671382613
  07. x 11 14 y 23 z 34  skor_komb=3.5510319632
  08. x 11 25 y 32 z 37  skor_komb=3.2859546418
  09. x 10 11 y 30 z 38  skor_komb=3.1775679882
  10. x 05 08 y 19 z 39  skor_komb=3.1243055239

Sacuvano: /Tesla/tesla_scalar_1.txt
"""



"""
source ~/tesla_env/bin/activate

Bitne verzije za tesla_env:

Paket	Verzija
python  3.11.13
numpy   2.2.6
scipy   1.15.3
pandas  3.0.3
matplotlib    3.10.9
k-Wave-python 0.6.2
pycharge      2.0.1
jax        0.10.1
jaxlib     0.10.1
jaxtyping  0.3.7
equinox    0.13.8
lineax     0.1.1
optimistix 0.1.0
ml-dtypes
(uz jax)
opencv-python 4.13.0.92
h5py          3.16.0
"""
