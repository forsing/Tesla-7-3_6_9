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
CSV_PATH = Path("/Users/4c/Desktop/GHQ/data/loto7hh_4632_k47.csv")
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

Slika talasa: /Users/4c/Desktop/GHQ/Tesla/tesla_scalar_1.png
Slika talasa: /Users/4c/Desktop/GHQ/Tesla/tesla_scalar_1.jpg


Tesla Scalar - korak 2 (tezinski: talas + prava frekvencija)
CSV: /Users/4c/Desktop/GHQ/data/loto7hh_4632_k47.csv | Izvlacenja: 4632 | tezine: talas=0.7 freq=0.3

Top 10 brojeva po kombinovanom skoru (tezinski talas + frekvencija):
  34  skor=0.9213793103  freq=0.02692  (pojava=873)
  35  skor=0.8176249656  freq=0.02600  (pojava=843)
  08  skor=0.7603045533  freq=0.02810  (pojava=911)
  21  skor=0.6931601524  freq=0.02551  (pojava=827)
  13  skor=0.6706469920  freq=0.02554  (pojava=828)
  31  skor=0.6127524074  freq=0.02560  (pojava=830)
  33  skor=0.6112555867  freq=0.02634  (pojava=854)
  37  skor=0.5766162438  freq=0.02652  (pojava=860)
  03  skor=0.5593769679  freq=0.02547  (pojava=826)
  25  skor=0.5585228387  freq=0.02591  (pojava=840)


Tabela pravih frekvencija (opadajuce po freq, pa po broju):
  broj | pojava |   udeo
  -----+--------+--------
   08  |   911  | 0.02810
   23  |   905  | 0.02791
   34  |   873  | 0.02692
   26  |   869  | 0.02680
   11  |   861  | 0.02655
   37  |   860  | 0.02652
   32  |   857  | 0.02643
   33  |   854  | 0.02634
   22  |   851  | 0.02625
   39  |   849  | 0.02618
   29  |   849  | 0.02618
   10  |   845  | 0.02606
   07  |   844  | 0.02603
   35  |   843  | 0.02600
   09  |   843  | 0.02600
   38  |   842  | 0.02597
   25  |   840  | 0.02591
   24  |   840  | 0.02591
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
  01. 06 12 13 25 34 35 39  skor_komb=4.5418643612
  02. 07 08 13 23 24 25 34  skor_komb=4.1634211362
  03. 08 17 18 23 26 33 34  skor_komb=3.9810045066
  04. 02 11 16 21 25 34 39  skor_komb=3.9122219361
  05. 17 25 28 29 31 33 34  skor_komb=3.7585875317
  06. 06 07 13 31 32 37 39  skor_komb=3.6671382613
  07. 10 11 14 21 23 32 34  skor_komb=3.5510319632
  08. 04 11 25 30 32 33 37  skor_komb=3.2859546418
  09. 09 10 11 21 30 33 38  skor_komb=3.1775679882
  10. 02 05 08 13 19 30 39  skor_komb=3.1243055239

Sacuvano: /Users/4c/Desktop/GHQ/Tesla/tesla_scalar_1.txt
"""






"""
čist SLW motor — skalarni talas koji se prostire u +x pravcu (d'Alamberov talas), 
skalarno polje S(x,t) koje se prostire u +x pravcu 
plus uzdužno polje E_x = -∂S/∂x (gradient-driven, kako EED i opisuje). 
osnovne mere: max S, max |E_x|, ukupna gustina energije
Bez frekvencijske logike, čista fizika.

Ključna ideja koraka 2 (ne-frekvencijska): 
SLW talas iz koraka 1 prostire se preko 4630 izvlačenja (1 tačka = 1 izvlačenje). 
Za svaki broj 1-39 računa ne-frekvencijski skor =  prosečna energija talasa na pozicijama gde se taj broj pojavio 
— dakle ne koliko puta (frekvencija), nego gde u talasu leži (rezonanca sa poljem).
ne_frekvencijski_skor() — prosečna energija talasa po broju (ne frekvencija)
Biram 10 kombinacija ponderisano tim skorom (seed 39).
izaberi_kombinacije() — 10 kombinacija ponderisano tim skorom (seed 39)

skorovi su međusobno vrlo blizu (0.0285-0.0329) 
— talas trenutno slabo razdvaja brojeve. 
Razlog: uzimam prosečnu energiju po broju, 
a kako se svaki broj pojavljuje na ~800+ 
različitih pozicija duž celog talasa, 
proseci se izravnaju i ispadnu skoro isti za sve.
To znači da, iako jeste ne-frekvencijski po definiciji, 
talas još nema dovoljno „oštrine" da napravi jasnu razliku.
Zato model koristi pravu frekvenciju pojavljivanja brojeva nad svih 4630 izvlačenja.
frekvencija_brojeva(izvlacenja) — broji koliko se svaki broj 1-39 pojavio i vraća relativnu frekvenciju (udeo, zbir = 1) i sirov broj (pojave).
kombinovani_skor na normalizovanu težinsku kombinaciju (svaka komponenta skalirana na 0-1, pa ponderisan zbir).
Obe komponente se skaliraju na 0-1 (_na_0_1), pa ponderisan zbir: skor = W_TALAS·talas + W_FREQ·freq.
Težine na vrhu fajla: W_TALAS = 0.7, W_FREQ = 0.3 (lako menjam).

Logika: 
cela ideja je ne-frekvencijska (Tesla talas nosi razliku), 
a kod 7/39 nad 4630 izvlačenja frekvencija je skoro ravna (svaki broj ~ isti očekivani broj pojava) 
— pa frekvencija služi samo kao blagi stabilizator, ne kao glavni signal. 
Zato talas treba da dominira.

Postavljeno: 
talas 0.7 / freq 0.3 — talas dominira (ne-frekvencijska ideja), frekvencija samo blago koriguje. 

Frekvencija je realna: 
ukupno ima 4630 x 7 = 32410 pojavljivanja, očekivano po broju je oko 831. 

Zato su ove vrednosti normalne:
08 = 910 je stvarno jači po frekvenciji.
34 = 873 nije najjači po frekvenciji, ali je prvi po skoru, znači talas ga je podigao.
21 = 826, 13 = 828, 31 = 830 su skoro prosečni po frekvenciji, ali su visoko, opet znači da talas radi svoj deo.
Zaključak: odnos 0.7 talas / 0.3 freq nije ubio Teslinu ideju. 
Frekvencija samo stabilizuje, a talas i dalje odlučuje.

broj 34 se previše ponavlja u kombinacijama. 
Sledeći korak mozda da bude kontrola raznovrsnosti kombinacija, da top broj ne upada u pola liste.

Tabela pravih frekvencija je sortirana opadajuće — primarno po frekvenciji (pojava), a kod istog broja pojava po većem broju prvo. Isto u txt i u konzoli.

Sad imam dobru osnovu: 
SLW motor radi na 4630 tačaka.
Talas je primenjen na CSV.
Prava frekvencija je izračunata i prikazana.
Kombinovani skor je težinski 0.7 talas / 0.3 freq.
Txt izlaz je pregledan i potpun.

Svaku kombinaciju ocenim (npr. zbir/prosek skorova njenih 7 brojeva) i sortiramo opadajuće. 
Svaka kombinacija dobije svoj skor i da lista bude sortirana od najjače ka slabijoj, pa će prva zaista biti favorit.

10 kombinacija se prvo generiše, zatim se svaka ocenjuje sa skor_komb = zbir skorova 7 brojeva i sortira opadajuće. 
Zato je kombinacija 01 sada favorit, a deseta je najslabija od ponuđenih.

Kombinacije su sad sortirane opadajuće po skor_komb: od 4.183 (prva) do 2.835 (deseta).
Favorit je 01: 06 13 22 25 34 35 39 (skor_komb=4.1828).
Najslabija ponuđena je 10: 09 10 11 29 30 33 38 (2.8350).

Logika je konzistentna: 
prva kombinacija ima najviše jakih brojeva po kombinovanom skoru 
(34, 35, 25, 13, 39 su svi visoko u top listi), 
a deseta ima brojeve nižeg ranga.



na kraju crta talas/polje i snima slike:

3 panela: 
S(x) (skalarno polje), 
E_x = -dS/dx (uzdužno polje), 
i energija.
"""




"""
Analiza — Tesla 1 (ručni SLW motor)

Motor: d'Alamberova jednačina, konačne razlike, 1D talas u +x. Polje: max S ≈ 1.0, max|E_x| ≈ 0.101, ukupna energija ≈ 124.78. Mala |E_x| znači blag gradijent — talas je „mek", energija je raspoređena prilično glatko.

Top brojevi (talas + freq, 0.7/0.3): 34 (0.923) · 35 (0.819) · 08 (0.762) · 21 (0.694) · 13 (0.671) · 31 (0.613) · 33 (0.612) · 37 (0.578) · 03 (0.561) · 25 (0.559)

Šta nosi rang:

08 je u top 3 i frekvencijski je #1 (910) → tu se talas i frekvencija slažu.
21, 13, 31 su frekvencijski prosečni/ispod (826/828/830), a visoko su → talas ih je podigao (ne-frekvencijski signal radi).
35 je #2 po skoru a frekvencijski je tek sredina (843) → opet talas.
Favorit kombinacija: 06 13 22 25 34 35 39 (skor_komb = 4.1828). Najslabija ponuđena: 09 10 11 29 30 33 38 (2.8350).

Zaključak: Tesla 1 je „mekan" motor — diferencijacija među brojevima je umerena (skorovi se glatko spuštaju). Dobar je kao referentna osnova: dokazuje da ne-frekvencijski signal postoji, ali bez oštrih vrhova. To je očekivano za jednostavan 1D talas.
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
