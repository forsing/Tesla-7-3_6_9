"""
SRBIN Nikola Tesla, za sva vremena, najveci naucnik sveta.

SERBIAN Nikola Tesla, for all time, the greatest scientist in the world.
"""


"""
Tesla_369_7.py  —  GRUPA 7: Tesla 7-harmonijski motor po pozicijama
"""


"""
Prosirenje Tesline 3-6-9 ideje na SEDAM harmonijskih slojeva:
  mnozioci = 3, 6, 9, 12, 15, 18, 21   (sedam slojeva)

Dva izlaza iz istog motora:
  1) SEDAM predikcija po pozicijama  -> po jedan broj za svaku poziciju 1..7
  2) JEDNA finalna kombinacija       -> spoj sedam pozicijskih brojeva

Mapiranje sloj -> pozicija (1:1):
  harmonik 3 -> pozicija 1 (najmanji broj) ... harmonik 21 -> pozicija 7 (najveci).

Opsezi po pozicijama se NE zadaju rucno (min/max). 
Posto su izvucene kombinacije u CSV-u rastuce sortirane (Num1<=...<=Num7), 
svaka pozicija ima svoju prirodnu empirijsku raspodelu, 
pa brojevi sigurno upadaju u svoj opseg.

S(x) = sloj sin(2*pi*m*baza*x) sa fokusnim omotacem (konvergencija u centru)
E_x  = -dS/dx
"""



import numpy as np

from Tesla_Scalar_1 import (
    SEED,
    W_TALAS,
    W_FREQ,
    CSV_PATH,
    OUTPUT_DIR,
    MIN_BROJ,
    MAX_BROJ,
    ucitaj_izvlacenja,
    glavne_mere,
    nacrtaj_polje,
)

OSNOVA = "tesla_369_7"

# Sedam Teslinih harmonijskih slojeva (3..21, korak 3).
MNOZIOCI_7 = (3, 6, 9, 12, 15, 18, 21)
BROJEVA_U_KOMBINACIJI = 7
BAZA_CIKLUSA = 21      # kao u grupi 5 (21 piramida -> 21 osnovnih ciklusa)
GAIN = 0.35            # correction_gain (Tesla 3-6-9 logika)
SIGMA_FOKUS = 0.16     # sirina fokusne konvergencije oko centra
K_SHRINK = 30          # koliko uzoraka treba da bi se prosek talasa "verovao"
                       # (retki ekstremi -> povuceni ka proseku, ne pobedjuju na sumu)


def sloj_talas(mnozilac, nx, baza_ciklusa=BAZA_CIKLUSA, gain=GAIN, sigma=SIGMA_FOKUS):
    """Vrati (x, S, E_x) za JEDAN harmonijski sloj sin(2*pi*m*baza*x) sa fokusom."""
    x = np.linspace(0.0, 1.0, nx)
    omotac = np.exp(-((x - 0.5) ** 2) / (2.0 * sigma ** 2))
    S = gain * omotac * np.sin(2.0 * np.pi * mnozilac * baza_ciklusa * x)
    m = np.max(np.abs(S))
    if m > 0:
        S = S / m
    E_x = -np.gradient(S, x[1] - x[0])
    return x, S, E_x


def zbirni_talas(nx, mnozioci=MNOZIOCI_7, baza_ciklusa=BAZA_CIKLUSA, gain=GAIN, sigma=SIGMA_FOKUS):
    """Sedam slojeva sabranih u jedan talas (za prikaz polja / sliku)."""
    x = np.linspace(0.0, 1.0, nx)
    omotac = np.exp(-((x - 0.5) ** 2) / (2.0 * sigma ** 2))
    stek = np.zeros(nx)
    for m in mnozioci:
        stek += np.sin(2.0 * np.pi * m * baza_ciklusa * x)
    S = gain * omotac * stek
    mx = np.max(np.abs(S))
    if mx > 0:
        S = S / mx
    E_x = -np.gradient(S, x[1] - x[0])
    return x, S, E_x


def _na_0_1(recnik):
    """Min-max na 0..1 unutar datog recnika (ravno na 0 ako su sve jednake)."""
    if not recnik:
        return {}
    v = np.array(list(recnik.values()), dtype=float)
    raspon = v.max() - v.min()
    if raspon <= 0:
        return {k: 0.0 for k in recnik}
    return {k: (recnik[k] - v.min()) / raspon for k in recnik}


def pozicijski_skor(izvlacenja, energija, poz, k_shrink=K_SHRINK):
    """Za datu poziciju (0..6): stabilan talasni skor i pojave po broju.

    talas[b] = shrinkage prosek energije na izvlacenjima gde je na toj poziciji bio b:
      (zbir_e[b] + k * global_mean) / (pojave[b] + k)
    Tako brojevi sa malo uzoraka (retki ekstremi) ne pobede na samo jednom pogotku,
    nego se povuku ka prosecnoj energiji te pozicije.
    """
    zbir, pojave = {}, {}
    for i, red in enumerate(izvlacenja):
        b = red[poz]
        zbir[b] = zbir.get(b, 0.0) + float(energija[i])
        pojave[b] = pojave.get(b, 0) + 1
    global_mean = float(np.mean(energija))
    talas = {b: (zbir[b] + k_shrink * global_mean) / (pojave[b] + k_shrink) for b in zbir}
    return talas, pojave


def kombinovani_pozicijski(talas, pojave, w_talas=W_TALAS, w_freq=W_FREQ):
    """Tezinski skor po poziciji: normalizovan talas + normalizovana pozicijska frekvencija."""
    t = _na_0_1(talas)
    f = _na_0_1({b: float(pojave[b]) for b in pojave})
    return {b: w_talas * t[b] + w_freq * f[b] for b in talas}


def predikcije_po_pozicijama(izvlacenja):
    """Vrati listu od 7 dict-ova (po poziciji) + izabrane brojeve uz dedupe."""
    n = len(izvlacenja)
    rezultati = []
    izabrani = []
    zauzeti = set()

    for poz, m in enumerate(MNOZIOCI_7):
        _, S, E_x = sloj_talas(m, n)
        energija = 0.5 * (S ** 2 + E_x ** 2)
        talas, pojave = pozicijski_skor(izvlacenja, energija, poz)
        skor = kombinovani_pozicijski(talas, pojave)
        poredak = sorted(skor.items(), key=lambda kv: kv[1], reverse=True)

        # Dedupe: uzmi najjaci broj koji jos nije zauzet ranijom pozicijom.
        izbor = None
        for b, _ in poredak:
            if b not in zauzeti:
                izbor = b
                break
        if izbor is None:  # teorijski; svi zauzeti
            izbor = poredak[0][0]
        zauzeti.add(izbor)
        izabrani.append(izbor)

        rezultati.append({
            "pozicija": poz + 1,
            "harmonik": m,
            "izbor": izbor,
            "skor": skor,
            "pojave": pojave,
            "poredak": poredak,
        })

    return rezultati, izabrani


def main():
    izvlacenja = ucitaj_izvlacenja()
    n = len(izvlacenja)

    # Zbirni 7-harmonijski talas (samo za mere i sliku polja).
    x, S, E_x = zbirni_talas(n)
    mere = glavne_mere(S, E_x)

    rezultati, izabrani = predikcije_po_pozicijama(izvlacenja)
    finalna = sorted(izabrani)

    png, jpg = nacrtaj_polje(x, S, E_x, osnova=OSNOVA)

    txt_path = OUTPUT_DIR / f"{OSNOVA}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Tesla Scalar - GRUPA 7 / 7 harmonika po pozicijama (3-6-9-12-15-18-21)\n")
        f.write(f"CSV: {CSV_PATH}\n")
        f.write(f"Izvlacenja: {n} | Seed: {SEED} | tezine: talas={W_TALAS} freq={W_FREQ}\n")
        f.write(f"Mnozioci: {MNOZIOCI_7} | baza_ciklusa={BAZA_CIKLUSA} gain={GAIN} sigma={SIGMA_FOKUS}\n")
        f.write(f"max S: {mere['max_S']:.10f} | max |E_x|: {mere['max_abs_E_x']:.10f}\n\n")

        f.write("1) SEDAM predikcija po pozicijama (sloj -> pozicija):\n")
        for r in rezultati:
            b = r["izbor"]
            f.write(
                f"  poz {r['pozicija']}  (harmonik {r['harmonik']:2d})  ->  broj {b:02d}"
                f"   skor={r['skor'][b]:.10f}  (pojava na poziciji={r['pojave'][b]})\n"
            )
            top3 = ", ".join(f"{bb:02d}({r['skor'][bb]:.4f})" for bb, _ in r["poredak"][:3])
            f.write(f"        top kandidati: {top3}\n")

        f.write("\n2) FINALNA kombinacija (spoj sedam pozicija):\n")
        f.write("  " + " ".join(f"{b:02d}" for b in finalna) + "\n")

        f.write("\nDetaljno po poziciji (svi kandidati po skoru):\n")
        for r in rezultati:
            f.write(f"\n  Pozicija {r['pozicija']} (harmonik {r['harmonik']}):\n")
            for b, v in r["poredak"]:
                f.write(f"    {b:02d}  skor={v:.10f}  (pojava={r['pojave'][b]})\n")

        f.write("\nSlike talasa/polja:\n")
        f.write(f"  PNG: {png}\n")
        f.write(f"  JPG: {jpg}\n")

    # --- konzola ---
    print()
    print("Tesla Scalar - GRUPA 7 / 7 harmonika po pozicijama (3-6-9-12-15-18-21)")
    print(f"CSV: {CSV_PATH} | Izvlacenja: {n} | tezine: talas={W_TALAS} freq={W_FREQ}")
    print(f"max S: {mere['max_S']:.10f} | max |E_x|: {mere['max_abs_E_x']:.10f}")

    print("\n1) SEDAM predikcija po pozicijama:")
    for r in rezultati:
        b = r["izbor"]
        top3 = ", ".join(f"{bb:02d}({r['skor'][bb]:.4f})" for bb, _ in r["poredak"][:3])
        print(
            f"  poz {r['pozicija']} (harmonik {r['harmonik']:2d}) -> broj {b:02d}"
            f"  skor={r['skor'][b]:.6f}  (pojava={r['pojave'][b]})   [top: {top3}]"
        )

    print("\n2) FINALNA kombinacija (spoj sedam pozicija):")
    print("  " + " ".join(f"{b:02d}" for b in finalna))

    print(f"\nSacuvano: {txt_path}")
    print()


if __name__ == "__main__":
    main()



"""
Slika talasa: /Tesla/tesla_369_7.png
Slika talasa: /Tesla/tesla_369_7.jpg

Tesla Scalar - GRUPA 7 / 7 harmonika po pozicijama (3-6-9-12-15-18-21)
CSV: /data/loto7hh_4632_k47.csv| Izvlacenja: 4632 | tezine: talas=0.7 freq=0.3
max S: 1.0000000000 | max |E_x|: 1966.6524828516

1) SEDAM predikcija po pozicijama:
  poz 1 (harmonik  3) -> broj 02  skor=0.844380  (pojava=681)   [top: 02(0.8444), 03(0.7523), 20(0.7015)]
  poz 2 (harmonik  6) -> broj 18  skor=0.765743  (pojava=88)   [top: 18(0.7657), 09(0.7385), 08(0.6495)]
  poz 3 (harmonik  9) -> broj 22  skor=0.864286  (pojava=162)   [top: 22(0.8643), 13(0.8384), 12(0.8049)]
  poz 4 (harmonik 12) -> broj  x  skor=0.825517  (pojava=162)   [top: 13(0.8255), 24(0.7473), 25(0.7277)]
  poz 5 (harmonik 15) -> broj  y  skor=0.904477  (pojava=232)   [top: 21(0.9045), 26(0.8667), 16(0.7954)]
  poz 6 (harmonik 18) -> broj  z  skor=0.863077  (pojava=213)   [top: 26(0.8631), 31(0.7419), 37(0.7275)]
  poz 7 (harmonik 21) -> broj 33  skor=0.792689  (pojava=263)   [top: 33(0.7927), 31(0.6835), 39(0.6752)]

2) FINALNA kombinacija (spoj sedam pozicija):
  02 x 18 y 22 z 33

Sacuvano: /Tesla/tesla_369_7.txt
"""



"""
Analiza Tesla_369_7.py
Tesla_369_7.py je novi model koji ne pravi klasičnu 7/39 kombinaciju iz jedne opšte liste brojeva, nego radi pozicijski: svaka od 7 loto pozicija dobija svoj harmonijski sloj.

Glavna ideja je:

3  -> pozicija 1
6  -> pozicija 2
9  -> pozicija 3
12 -> pozicija 4
15 -> pozicija 5
18 -> pozicija 6
21 -> pozicija 7
To je dobra nadogradnja u odnosu na stari 3-6-9 model, jer sada broj harmonika odgovara broju loto brojeva.

Model koristi harmonijsku superpoziciju:

MNOZIOCI_7 = (3, 6, 9, 12, 15, 18, 21)
Svaki sloj pravi svoj talas:

S = gain * omotac * np.sin(2.0 * np.pi * mnozilac * baza_ciklusa * x)
E_x = -np.gradient(S, x[1] - x[0])

sinusni harmonik
fokusni Gaussov omotač oko centra
normalizacija talasa
izvod E_x, kao uzdužni/longitudinalni proxy
energiju polja 0.5 * (S² + E_x²)

Za svaku poziciju:
Uzme odgovarajući harmonik.
Napravi poseban talas.
Izračuna energiju talasa.
Gleda samo tu jednu CSV poziciju, npr. Num3 za treću poziciju.
Računa skor kandidata samo među brojevima koji se realno javljaju na toj poziciji.
Bira najjači broj koji još nije zauzet.
Nema min/max ograničenja. Opseg dolazi direktno iz CSV-a, jer su Num1..Num7 već rastuće sortirani.

Najpametniji deo je K_SHRINK = 30.

Bez toga bi broj koji se pojavio samo jednom mogao da dobije ogromnu prosečnu energiju i pobedi slučajno. 
Zato kod koristi stabilizaciju:
(zbir_e[b] + k * global_mean) / (pojave[b] + k)
To znači: 
ako broj ima malo pojava, njegov talasni skor se povlači ka proseku. 
Ako ima mnogo pojava, model mu više veruje.

To je dobra zaštita od šuma, a nije ručno ograničavanje opsega.

Finalni skor po poziciji je:
0.7 * talas + 0.3 * pozicijska_frekvencija
Ovo nije opšta frekvencija broja u celom CSV-u, nego frekvencija na toj poziciji. 
To je ispravno za ovaj model.

Dedupe
Dedupe je jednostavan i dobar:

if b not in zauzeti:
    izbor = b
Ako je isti broj već izabran na ranijoj poziciji, 
sledeća pozicija uzima sledećeg najboljeg kandidata. 
Tako finalna kombinacija uvek ima 7 različitih brojeva.

02 x 18 y 22 z 33
Pozicijski gledano, rezultat ima logičan oblik:

prvi broj je mali: 02
sredina je oko 18-22
kraj ide ka većem broju: 33
To pokazuje da empirijska pozicijska raspodela radi kako treba. 
Posle shrinkage popravke više ne pobeđuju ekstremi sa jednom pojavom, 
nego brojevi koji imaju realnu podršku u CSV-u.

Zaključak: Tesla_369_7.py je dobar poseban model, različit od agregatora i od starog 3-6-9 modela. Njegova snaga je baš u tome što ne bira 7 brojeva iz jedne opšte mase, nego svaki broj izvlači iz svoje pozicione harmonike.
"""



"""
7 harmonijskih slojeva 3-6-9-12-15-18-21, mapiranih 1:1 na pozicije 1..7 (harmonik 3 → najmanji broj … harmonik 21 → najveći).
Kolone Num1..Num7 u CSV-u rastuće sortirane, svaka pozicija ima svoju prirodnu empirijsku raspodelu, pa brojevi sami upadaju u opseg.
Skor po poziciji = 0.7·talas + 0.3·pozicijska_frekvencija, gde je talasni prosek stabilizovan shrinkage-om (K_SHRINK=30) da retki ekstremi (1 pojava) ne pobede na šumu.
Dedupe: pozicije se biraju redom, ako je broj zauzet uzima se sledeći najjači za tu poziciju.

7 predikcija po pozicijama
jedna finalna kombinacija (spoj sedam pozicija)

poz 1 (h 3)  -> 02   (pojava=681)
poz 2 (h 6)  -> 18   (pojava=88)
poz 3 (h 9)  -> 22   (pojava=162)
poz 4 (h12)  -> x   (pojava=162)
poz 5 (h15)  -> y   (pojava=232)
poz 6 (h18)  -> z   (pojava=213)
poz 7 (h21)  -> 33   (pojava=263)
FINALNA: 02 x 18 y 22 z 33
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
