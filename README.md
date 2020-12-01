# Entwicklung einer KI, die Texte verstehen kann

Diese Repo gehört zu den doiit-Beiträgen von [kindustrii](https://www.kindustrii.de).
Hier zeige ich Schritt für Schritt auf, wie eine KI bzw. ein neuronales Netz erstellt wird, das in der Lage ist Texte zu verarbeiten.

Eine KI die Texte verarbeitet, kann z.B. für die folgenden Aufgaben verwendet werden:
- Erkennen um welches Thema es sich in dem Text handelt
- Prüfen, ob zwei unterschiedliche Texte das Gleiche Thema behandeln
- Anhand des Textes erkennen, von welchem Autor dieser Text verfasst wurde
- Das Umwandeln der Texte von einer in eine andere Sprache 
- Stimmungsanalyse, also erkennen welche Stimmung der Text vermittelt
- Beim Schreiben eines Textes können Wortvorhersagen getroffen werden
- Prüfen von Texten auf die Einhaltung von Grammatikregeln

Diese Art von KI ist heute schon weit verbreitet. Sie wird z.B. für Chat-Bots, die Google Suche oder in Textverarbeitungsprogrammen verwendet.

In diesem Beispiel hier werden wir eine KI entwickeln, die eine Filmbewertung analysieren soll. Bei diesen Filmbewertungen bzw. Rezessionen handelt es sich um echte Filmbewertungen, die von Menschen erstellt wurden. Die KI wird anhand des Textes erkennen, ob die Rezession positiv oder negativ ist. In anderen Worten: Die KI erkennt, ob der Film dem Verfasser der Rezession gefallen hat oder nicht.

**Den Code inkl. Erläuterungen dazu und die eigentliche KI-Entwicklung findest Du hier:** [doiit-text-klassifizierung-rnn](https://github.com/kindustrii/doiit-text-klassifizierung-rnn/blob/main/doiit_text_klassifizierung_mit_rnn.ipynb)

## 1. Aufgabenstellung und Problemanalyse 

Wenn die KI entwickelt und trainiert ist, übergeben wir an sie eine Filmbewertung, die von ihr verarbeitet und entsprechend als positiv oder negativ bewertet werden soll. Damit sie das auch richtig macht, werden wir das Neuronale Netz trainieren müssen. Im Training erhält sie von uns 25.000 positive und negative Rezessionen vorgelegt. Es handelt sich hier um das Supervised Learning.

Das Verarbeiten, bzw. in unserem Fall das Klassifizieren von Texten bringt gewisse Besonderheiten und Herausforderungen mit, die eine KI bewerkstelligen muss. 

### Reihenfolge der Wörter

So ist zum Beispiel die Reihenfolge der Wörter in einem Satz von sehr hoher Bedeutung. Hier ein Beispiel:
1. *Mann beißt Hund*
2. *Hund beißt Mann*

Beide Sätze enthalten exakt dieselbe Anzahl von Wörtern und verwenden auch dieselben Wörter. Der einzige Unterschied ist, dass die Wörter *Mann* und *Hund* vertauscht sind. Die KI muss also die Reihenfolge der Wörter und der Sätze beachten.

### Bedeutung einzelner Wörter

Ein Wort kann auf zwei Arten unterschiedliche Bedeutung haben:
1. Unabhängig vom Kontext hat das Wort verschiedene Bedeutungen im Sprachgebrauch. Das Wort *Decke* kann einmal die horizontal angebrachte Wand über Deinem Kopf sein, aber auch das große Tuch unter dem Du Dich nachts im Bett vor der Kälte versteckst.
2. Je nach Kontext können die Wörter etwas anderes bedeuten. *Wenig Punkte* in Flensburg zu haben ist etwas Gutes, wohingegen *wenig Punkte* in einer Prüfung zu haben nicht unbedingt etwas Gutes ist.

### Verschiedene Aussagen mit gleicher Bedeutung

Schaue Dir die folgenden Aussagen an:
1. *Mir ist kalt.*
2. *Ich friere.*
3. *Mach die Heizung an.*
4. *Ich muss mir etwas Wärmeres anziehen.*

Die vier oberen Aussagen bedeuten alle das Gleiche. Beachte, dass sie alle unterschiedlich lang sind und bis auf die Wörter *ich* und *mir* keine gleichen Wörter verwenden.

Es gibt natürlich noch weitere Aspekte, die bei der Verarbeitung von Text zum Hindernis werden könnten:
- Falsche Rechtschreibung kann ein Wort unkenntlich machen, oder sogar ein ganz anderes Wort daraus machen
- Grammatik hat auch einen großen Einfluss auf die Bedeutung eines Satzes
- Abkürzungen, Fachbegriffe, Synonyme, Slang, Wörter die es gar nicht gibt (*Froschhaarpinsel*), uvm.

### Die Daten bestehen aus Buchstaben, Wörtern und Sätzen 

Ein Neuronales Netz, das Bilder bzw. Fotos analysiert, erhält Daten die aus Zahlen bestehen. Ein Bild mit z.B. der Auslösung 512x512 besteht aus 512 Zeilen mit je 512 Spalten Pixel. Das sind insgesamt 262.144 Pixel. Wobei es sich bei einem Pixel um einen Zahlenwert zwischen 0 und 255 handelt. Jede dieser Zahlen repräsentiert eine Farbe.

Da Neuronale Netze mit Zahlen arbeiten, müssen die Wörter erstmal in Zahlen gewandelt werden. Was sich einfach anhört, ist in Wirklichkeit eine Wissenschaft für sich.

### Bei Texten handelt es sich um Zeitreihen bzw. um Sequenzen

Wenn das Neuronale Netz einen Text der aus 6 einzelnen Sätzen besteht verarbeitet und ist am vierten Satz angekommen, dann muss es die vorherigen 3 Sätze berücksichtigen, um den vierten Satz richtig zu verstehen. Dafür wird eine besondere Architektur des Neuronalen Netzes benötigt, die über eine Art Gedächtnis verfügt.

## 2. Lösungsidee

Da es sich hier um einen doiit-Beitrag handelt, werde ich die theoretische Beschreibung der Lösung nur sehr kurz halten und konzentriere mich eher auf den praktischen Teil. Mehr theoretische Details zu den einzelnen Elementen der Lösung findest Du in den techniik-Beiträgen auf [kindustrii](https://www.kindustrii.de).

Die ganzen Herausforderungen sind zum Glück nicht neu, sodass sich einige sehr elegante Lösungen dazu finden lassen. Wir werden das Rad nicht neu erfinden und greifen auf die bereits vorhandenen Lösungsansätze zurück.

### Text in Zahlen umwandeln

Als allererstes müssen wir den Text in ein Datenformat konvertieren, mit wir anständig arbeiten können. Wie es die Überschrift schon verrät werden wir den Text in Zahlen umwandeln. Uns stehen insgesamt 50.000 Filmbewertungen als Text zur Verfügung, in denen 121.894 verschiedene Wörter auftauchen. Wir gehen folgendermassen  vor:

1. Alle 50.000 Bewertungen durchlaufen und dabei alle unterschiedlichen Wörter in eine Vokabelliste eintragen
2. Jedem Wort in der Vokabelliste eine eindeutige Zahl zuweisen
3. Wieder alle 50.000 Filmbewertungen durchgehen und dabei jedes Wort anhand der Vokabelliste gegen die entsprechende Zahl austauschen

Als Code sieht das ganze dann wie folg aus.

```python
# 5 Beispielbewertungen in Textform anzeigen
for bewertung,markierung in dataset['train'].take(5):
  print('Bewertung:  {}'.format(bewertung.numpy()))
```
```python
Bewertung:  b"This was an absolutely terrible movie. Don't be lured in by Christopher Walken or Michael Ironside. Both are great actors, but this must simply be their worst role in history. Even their great acting could not redeem this movie's ridiculous storyline. This movie is an early nineties US propaganda piece. The most pathetic scenes were those when the Columbian rebels were making their cases for revolutions. Maria Conchita Alonso appeared phony, and her pseudo-love affair with Walken was nothing but a pathetic emotional plug in a movie that was devoid of any real meaning. I am disappointed that there are movies like this, ruining actor's like Christopher Walken's good name. I could barely sit through it."

Bewertung:  b'I have been known to fall asleep during films, but this is usually due to a combination of things including, really tired, being warm and comfortable on the sette and having just eaten a lot. However on this occasion I fell asleep because the film was rubbish. The plot development was constant. Constantly slow and boring. Things seemed to happen, but with no explanation of what was causing them or why. I admit, I may have missed part of the film, but i watched the majority of it and everything just seemed to happen of its own accord without any real concern for anything else. I cant recommend this film at all.'

Bewertung:  b'Mann photographs the Alberta Rocky Mountains in a superb fashion, and Jimmy Stewart and Walter Brennan give enjoyable performances as they always seem to do. <br /><br />But come on Hollywood - a Mountie telling the people of Dawson City, Yukon to elect themselves a marshal (yes a marshal!) and to enforce the law themselves, then gunfighters battling it out on the streets for control of the town? <br /><br />Nothing even remotely resembling that happened on the Canadian side of the border during the Klondike gold rush. Mr. Mann and company appear to have mistaken Dawson City for Deadwood, the Canadian North for the American Wild West.<br /><br />Canadian viewers be prepared for a Reefer Madness type of enjoyable howl with this ludicrous plot, or, to shake your head in disgust.'

Bewertung:  b'This is the kind of film for a snowy Sunday afternoon when the rest of the world can go ahead with its own business as you descend into a big arm-chair and mellow for a couple of hours. Wonderful performances from Cher and Nicolas Cage (as always) gently row the plot along. There are no rapids to cross, no dangerous waters, just a warm and witty paddle through New York life at its best. A family film in every sense and one that deserves the praise it received.'

Bewertung:  b'As others have mentioned, all the women that go nude in this film are mostly absolutely gorgeous. The plot very ably shows the hypocrisy of the female libido. When men are around they want to be pursued, but when no "men" are around, they become the pursuers of a 14 year old boy. And the boy becomes a man really fast (we should all be so lucky at this age!). He then gets up the courage to pursue his true love.'
```
```python
# Einen Encoder erstellen
encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=None)
# Den Encoder alle Filmbewertungen durchlaufen und Wörter sammeln lassen
encoder.adapt(train_data.map(lambda text, label: text))
```
```python
# die gleichen 5 Beispielbewertungen wie oben, mit dem Encoder in Zahlen umwandeln und anzeigen
for bewertung,markierung in dataset['train'].take(5):
  print('Bewertung:  {}'.format(encoder(bewertung)))
```
```python
Bewertung:  [   11    14    34   412   384    18    90    28 10690     8    33  1322
  3560    42   487 11832   191    24    85   152    19    11   217   316
    28    65   240   214     8   489    54    65    85   112    96    22
  5596    11    93   642   743    11    18     7    34   394  9522   170
  2464   408     2    88  1216   137    66   144    51     2 42477  7558
    66   245    65  2870    16 18297  2860 22314 21154  1426  5050     3
    40 76666  1579    17  3560    14   158    19     4  1216   891  8040
     8     4    18    12    14  4059     5    99   146  1241    10   237
   704    12    48    24    93    39    11  7339   152    39  1322 14109
    50   398    10    96  1155   851   141     9]
Bewertung:  [   10    26    75   617     6   776  2355   299    95    19    11     7
   604   662     6     4  2129     5   180   571    63  1403   107  2410
     3  3905    21     2 38136     3   252    41  4781     4   169   186
    21    11  4259    10  1507  2355    80     2    20    14  1973     2
   114   943    14  1740  1300   594     3   356   180   446     6   596
    19    17    57  1775     5    49    14  4002    98    42   134    10
   934    10   194    26  1026   171     5     2    20    19    10   284
     2  2065     5     9     3   279    41   446     6   596     5    30
   200 36551   201    99   146  4525    16   229   329    10   175   368
    11    20    31    32]
Bewertung:  [ 4816  6359     2 31944  4995  3927     8     4   895  1620     3  1940
  1346     3  2300  9228   192   736   347    15    35   203   289     6
    82    13    13    19   209    21   355     4 27304  1018     2    83
     5  4001   541 15818     6 35355   520     4 12147   414     4 12147
     3     6 28065     2  1355   520    92 41029  8114     9    46    21
     2  1889    16  1158     5     2   508    13    13   158    54  2501
  7226    12   562    21     2  2263   506     5     2  3678   299     2
 23435  1958  3292   436  4816     3  1151   945     6    26  4141  4001
   541    16 23855     2  2263  2417    16     2   314  1283 26319    13
  2263   722    28  2772    16     4 19297  3029   584     5   736 18586
    17    11  2880   114    42     6  4500   123   426     8  6676]
Bewertung:  [   11     7     2   238     5    20    16     4  8730  2634  2650    51
     2   343     5     2   188    69   138  1431    17    30   200   978
    15    23 19786    78     4   196 31869     3 21794    16     4   365
     5   612   374   347    36  6789     3  5485  1908    15   203  8801
  3452     2   114   354    48    24    57 24803     6  2049    57  1776
  3779    41     4  2410     3  1888 80208   141   155   769   119    31
    30   115     4   222    20     8   168   278     3    29    12   985
     2  2858     9  1965]
Bewertung:  [   15   377    26  1041    32     2   362    12   138  2483     8    11
    20    24   639   412  1438     2   114    53 11097   265     2 11546
     5     2   653 15483    51   346    24   184    35   178     6    28
  7030    19    51    57   346    24   184    35   396     2 24815     5
     4  2606   336   165   444     3     2   444   440     4   133    63
   828    72   139    32    28    38  2086    31    11   582    27    92
   202    58     2  3086     6  6546    25   281   116]
```
