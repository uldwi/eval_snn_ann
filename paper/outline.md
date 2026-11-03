# Artikelentwurf: Effiziente SNN-Inferenz auf Mikrocontrollern

> Arbeitsstand: Die hier dokumentierte Messwerttabelle stammt aus dem neuesten, vom Autor bereitgestellten Lauf und ist vorläufig. Messprotokoll, Plattformzuordnung und einzelne Ausreißer/fehlgeschlagene Läufe sind noch zu verifizieren. Änderungen an Ergebnissen zuerst hier dokumentieren und in `paper.tex` nachziehen.

## Arbeitstitel
- Spikende neuronale Netze für energieeffiziente Inferenz auf heterogenen Mikrocontrollern
- Arbeitssprache der ersten Fassung: Deutsch; englische Fassung/Titel später festlegen.

## Kernaussage und Forschungsfragen
- Ziel: prüfen, ob SNN für unterschiedliche Aufgaben auf Mikrocontrollern gegenüber vergleichbaren ANN Laufzeit- und/oder Energiegewinne erzielen.
- Konfigurierte Plattformen: ESP32-C6 (RISC-V, 160 MHz) und ESP32-S3 (Xtensa LX7, 240 MHz); Zuordnung der hier berichteten Läufe zu Boards muss noch bestätigt werden.
- Aufgaben umfassen MNIST/FashionMNIST/CIFAR-10, Keyword Spotting und PhysioNet EEG Motor Imagery.
- RQ1: Wie unterscheiden sich Genauigkeit, Laufzeit und Energie pro Inferenz von SNN und ANN?
- RQ2: Sind Effizienzvorteile aufgaben- und implementationsabhängig?
- RQ3: Welche Abwägungen zwischen Genauigkeit, Latenz, Energie und Spike-/SynOps-Aufwand treten auf?
- Vorläufige Beobachtung: SNN sind in mehreren berichteten Paaren schneller und energieärmer, während die Modellgenauigkeit teils geringer ist; andere Aufgaben zeigen keinen SNN-Vorteil.
- Effizienzclaims sind auf die konkret dokumentierten Messläufe zu beschränken.

## Abstract
- Motivation: Edge-Geräte begrenzen Energie, Laufzeit und Speicher.
- Methode: ANN/SNN für mehrere Bild-, Audio- und Biosignalaufgaben; Export und Messung auf ESP32-Mikrocontrollern mit Joulescope.
- Messumfang der vorliegenden Tabelle: 16 Einträge, darunter Varianten/Backends und fehlgeschlagene bzw. nicht vollständig gemessene Läufe.
- Vorläufige Hauptergebnisse: bei MNIST, FashionMNIST, CIFAR-10, EEGMI und einem KWS-CNN geringere SNN-Energie; Transformer-KWS und weitere Fälle zeigen Gegenbeispiele.
- Genauigkeit und Messwerte im Abstract erst nach Bereinigung der Modell-/Backend-Zuordnung zusammenfassen.

## 1. Einleitung
- Edge-Inferenz verlangt niedrige Energie bei akzeptabler Genauigkeit und Latenz.
- SNN verwenden zeitliche Spike-Aktivität; auf konventioneller MCU-Hardware muss deren rechnerischer Vorteil empirisch überprüft werden.
- Ein Vergleich muss reale Aufgaben, konkrete Deployment-Backends und physische Messungen umfassen.
- Beitrag: empirischer ANN/SNN-Vergleich über verschiedene Aufgabentypen anhand von Gerätegenauigkeit, Laufzeit, Energie, Modellgröße und – wo verfügbar – Spike-/SynOps-Metriken.
- Forschungsfragen wie oben.

## 2. Hintergrund und verwandte Arbeiten
- ANN- und SNN-Inferenz, LIF/IF-Neuronen, zeitliche Kodierung und Spike-Auswertung.
- Sparse Aktivität versus tatsächliche Rechen- und Energiekosten auf digitalen MCUs.
- TinyML und Energie-Messmethoden.
- Primärliteratur, konkrete Implementierungsbezüge und Zitierstil: noch ergänzen.

## 3. Methodik
### 3.1 Aufgaben und Datensätze
- MNIST und FashionMNIST: Bildklassifikation.
- CIFAR-10: anspruchsvollere Bildklassifikation.
- Speech Commands: Keyword Spotting; Tabelle enthält sowohl Compact/Spiking-Transformer als auch CNN-Modelle.
- PhysioNet EEGMI: Biosignal-/Motor-Imagery-Klassifikation mit zeitlichen und räumlichen Faltungen.
- Split, Vorverarbeitung, Anzahl Testfälle und Klassenverteilung pro Lauf genau dokumentieren.

### 3.2 Modelle und Training
- ANN/SNN-Paare soweit möglich mit ähnlicher Architektur und Modellgröße vergleichen.
- CNN-Paare verwenden laut Ergebnisbezeichnungen ähnliche Kanalstrukturen; SNN-Ausführung und Zeitschritte anhand Checkpoints/Konfiguration bestätigen.
- Speech-Commands-Transformer umfasst Compact ANN, Spike-Driven SNN und Spiking SNN.
- Trainingseinstellungen, Zufalls-Seeds, Modellcheckpoint, Encoding, Zeitschritte und Optimierungen je Messlauf nachtragen.

### 3.3 Hardware und Software
- Projektkonfiguration nennt ESP32-C6 (RISC-V, 160 MHz, 512 KB SRAM) und ESP32-S3 (Xtensa LX7, 240 MHz, 512 KB SRAM).
- Ergebnisüberschrift nennt ESP32, aber die konkrete Boardzuordnung pro Tabellenlauf fehlt noch.
- Backend-Zuordnung pro Zeile, Firmware-/ESP-IDF-Version, Compilerflags und Versorgung nachtragen.
- Leistung/Energie wurde laut Projektworkflow über Joulescope erfasst; Geräte- und Abtastratendetails ergänzen.

### 3.4 Messprotokoll und Kennzahlen
- Berichtete Felder: Hostgenauigkeit, ESP32-Genauigkeit, Modellgröße, Zeit/Inferenz, Energie/Inferenz, mittlere Leistung sowie SNN-Spikes und SynOps.
- Testumfang laut Tabelle je nach Aufgabe 100 oder 500 Beispiele.
- Unklar, ob Laufzeit/Energie Vorverarbeitung, Initialisierung und Modellladung einschließen; Definition nachtragen.
- Wiederholungszahl, Streuung, Konfidenzintervalle, Leerlaufkorrektur und Messreihenfolge fehlen.
- `N/A`- und Nullwerte nicht als valide Effizienzwerte interpretieren, bis Ursache geklärt ist.

### 3.5 Analyse
- Paarweiser Vergleich pro Aufgabe, Architektur und Backend.
- Energie- und Latenzverhältnis SNN/ANN sowie Genauigkeitsdifferenz ausweisen.
- Spike-/SynOps-Zahlen nur für Läufe mit gültiger Instrumentierung verwenden.
- Fehler, Backend-Duplikate und nicht abgeschlossene Messungen separat kennzeichnen.

## 4. Ergebnisse
### 4.1 Überblick über die vorläufigen Messwerte
- Tabelle enthält die vom Autor bereitgestellten neuesten Werte; alle Angaben sind vorläufig.
- KWS-Transformer: Spike-Driven SNN bei 14.88 mJ und 136.84 ms gegenüber ANN bei 16.51 mJ und 153.69 ms, aber niedrigere Hostgenauigkeit (63.28% gegenüber 85.89%). Zweite ANN-Zeile sowie SNN-Variante mit 214,688 Bytes/375.94 ms sind Backend/Build-Zuordnung ungeklärt.
- MNIST: SNN 19.39 mJ/170.42 ms gegenüber ANN 89.31 mJ/795.53 ms; ESP32-Genauigkeit 97.6% gegenüber 99.0%.
- EEGMI: SNN 25.76 mJ/233.60 ms gegenüber ANN 37.49 mJ/332.20 ms; ähnliche Gerätegenauigkeit (65.8% gegenüber 66.8%).
- FashionMNIST: SNN 87.62 mJ/774.26 ms gegenüber ANN 349.18 mJ/2918.23 ms; SNN hat niedrigere Hostgenauigkeit, aber gemeldete Gerätegenauigkeit 90.0% gegenüber 91.6%.
- CIFAR-10: SNN 53.64 mJ/466.76 ms gegenüber ANN 101.76 mJ/849.85 ms; ähnlich niedrige Genauigkeit (51.0% gegenüber 53.2%). Zusätzlicher SNN-Lauf ohne Zeit-/Energiewerte.
- KWS-CNN: SNN 14.65 mJ/134.07 ms gegenüber ANN 16.71 mJ/152.75 ms, bei niedrigerer Host- und Gerätegenauigkeit.
- Die SNN-Energie ist bei diesen fünf CNN/Biosignal-Paaren niedriger; das belegt noch keine allgemeine Überlegenheit und muss mit Protokoll-/Backenddetails abgesichert werden.
- Gemeldete mittlere Leistung liegt häufig ähnlich; geringere Gesamtenergie korreliert hier wesentlich mit kürzerer Inferenzzeit.

### 4.2 Modellqualität und Effizienz
- Host- und Gerätegenauigkeit gemeinsam darstellen; Abweichungen nicht übergehen.
- MNIST, FashionMNIST, EEGMI, CIFAR-10, KWS-CNN zeigen in der Tabelle jeweils niedrigere SNN-Energie, jedoch teils Genauigkeitsverlust.
- KWS-Transformer zeigt gemischtes Bild: Spike-Driven Variante effizienter als ANN bei deutlich niedrigerer Hostgenauigkeit; Spiking Transformer hat 32.34 mJ und 299.82 ms.
- ECG/HAR/EEG-Auge/Synthese sind in dieser aktuellen Tabelle nicht enthalten und nicht als aktuelle Resultate aufführen.

### 4.3 Spike- und SynOps-Metriken
- Spike- und SynOps-Zahlen werden für mehrere SNN-Läufe berichtet.
- Ein SNN-Lauf mit 0 Spikes/0 SynOps bei 41.62 mJ/375.94 ms ist erklärungsbedürftig und darf nicht als Nullaktivität interpretiert werden.
- Zwei zusätzliche Läufe haben N/A-Werte; Ursache und Messstatus klären.

### 4.4 Plattform- und Backendvergleich
- Boardzuordnung und Backend für jede Zeile nachtragen, bevor separate ESP32-C6-/S3-Schlussfolgerungen formuliert werden.
- Doppelte ANN-Zeilen und abweichende Modellgrößen auf Build-/Quantisierungs-/Backendvarianten prüfen.

## 5. Diskussion
- Ergebnisse deuten auf mögliche SNN-Vorteile bei Laufzeit und Energie in mehreren Aufgaben, bei gleichzeitigen Genauigkeitskosten.
- Gemessene mittlere Leistung ist ähnlich; verkürzte Laufzeit kann die Energie/Inferenz senken.
- Digitale Ausführung, neuronale Implementierung und Backend beeinflussen reale SNN-Effizienz.
- Transformer-Ausreißer zeigen: Architektur-/Implementierungswahl kann den Vorteil umkehren.
- Keine Verallgemeinerung auf alle MCUs/Aufgaben ohne geklärte Plattformzuordnung und Wiederholungen.

## 6. Limitationen und Validität
- Vorläufige Tabelle ohne Streuungen und Wiederholungszahlen.
- ESP32-C6/S3-Zuordnung pro Lauf fehlt.
- Doppelte/abweichende Modelle, N/A- und Nullmetriken erfordern Bereinigung.
- Modellgenauigkeiten und Testumfänge unterscheiden sich; faire Paarbildung prüfen.
- Messung erfasst möglicherweise Board-/Systemverbrauch statt isolierter Modellenergie.
- Datensatz-Splits, Seeds, Vorverarbeitung, Firmware und Compilerstand noch dokumentieren.

## 7. Schlussfolgerung
- Vorläufig: mehrere SNN-Läufe erreichen geringere Energie und Laufzeit als ihre ANN-Gegenstücke.
- Vorteile müssen gemeinsam mit Genauigkeit, Aufgabe, Architektur und Backend bewertet werden.
- Endgültige Aussagen erst nach Bereinigung und Wiederholung der Messungen.

## Tabellen und Abbildungen
- Tabelle mit allen Rohwerten und Status/Backend/Board je Messlauf.
- Abbildung Energie pro Inferenz gegen Gerätegenauigkeit, getrennt nach Aufgabe und MCU.
- Abbildung relative Energie- und Latenzänderung SNN gegenüber ANN.
- Tabelle Messaufbau, Firmware, Compiler, MCU und Joulescope-Konfiguration.

## Reproduzierbarkeit und offene Punkte
- Rohmessdaten, Konfiguration, Checkpoints, Seeds und Auswertung archivieren.
- Board- und Backendzuordnung für jede Zeile.
- Ursache für Duplikate, Null-/N/A-Spikes und fehlende Laufzeit-/Energiewerte.
- Wiederholungszahl, Streuung, genaue Messgrenzen und Systemenergie-Korrektur.
- Literatur, Zielpublikation, Sprache und Zitierstil.
