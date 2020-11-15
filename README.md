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

In diesem Beispiel hier werden wir eine KI entwickeln, die eine Filmbewertung soll. Bei diesen Filmbewertungen bzw. Rezessionen handelt es sich um echte Filmbewertungen, die von Menschen erstellt wurden. Die KI wird anhand des Textes erkennen, ob die Rezession positiv oder negativ ist. In anderen Worten: Die KI erkennt, ob der Film dem Verfasser der Rezession gefallen hat oder nicht.

## Welche Hindernisse sind zu bewältigen?

Wenn die KI hier entwickelt und trainiert ist, bekommt sie eine Filmrezension, die sie noch nie gesehen hat und soll uns sagen, ob es eine positive oder native Rezession ist. Damit sie das auch richtig macht, werden wir das Neuronale Netz trainieren müssen. Im Training erhält sie von uns viele positive und negative Rezessionen vorgelegt. Es handelt sich hier um das Supervised Learning.

Das Verarbeiten, bzw. in unserem Fall das Klassifizieren von Texten bringt gewisse Besonderheiten bzw. Herausforderungen mit, die eine KI bewerkstelligen muss. 

### Reihenfolge der Wörter

So ist zum Beispiel die Reihenfolge der Wörter in einem Satz von sehr hoher Bedeutung. Hier ein Beispiel:
1. *Mann beißt Hund*
2. *Hund beißt Mann*

Beide Sätze enthalten exakt dieselbe Anzahl von Wörtern und verwenden auch dieselben Wörter. Der einzige Unterschied ist, dass die Wörter *Mann* und *Hund* vertauscht sind. Die KI muss also die Reihenfolge der Wörter und der Sätze beachten.

### Bedeutung einzelner Wörter

Ein Wort kann auf zwei Arten unterschiedliche Bedeutung haben:
1. Unabhängig vom Kontext hat das Wort verschiedene Bedeutungen im Sprachgebrauch. Das Wort *Decke* kann einmal die horizontal angebrachte Wand über Deinem Kopf sein, aber auch das große Tuch unter dem Du Dich nachts im Bett vor der Kälte versteckst.
2. Je nach Kontext können die Wörter etwas anderes bedeuten. *Wenig Punkte* in Flensburg zu haben ist etwas Gutes, wohingegen *wenig Punkte* in der Prüfung zu haben nicht unbedingt etwas Gutes ist.

### Verschiedene Aussagen mit gleicher Bedeutung

Schaue Dir die folgenden Aussagen an:
1. *Mir ist kalt.*
2. *Ich friere.*
3. *Mach die Heizung an.*
4. *Ich muss mir etwas Wärmeres anziehen.*

Die vier oberen Aussagen bedeuten alle das Gleiche. Beachte, dass sie alle unterschiedlich lang sind und bis auf die Wörter *ich* und *mir* keine gleichen Wörter verwenden.

Es gibt natürlich noch weitere Aspekte, die bei der Verarbeitung von Text zum Hindernis werden könnten:
- Falsche Rechtschreibung kann ein Wort unkenntlich machen, oder die sogar ein ganz anderes Wort daraus machen
- Grammatik hat auch großen Einfluss auf die Bedeutung eines Satzes
- Abkürzungen, Fachbegriffe, Synonyme, Slang, Wörter die es gar nicht gibt (*Froschhaarpinsel*), uvm.
