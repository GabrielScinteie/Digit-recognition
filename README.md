# Digit Recognition WebApp using Pytorch and Flask

## Descriere
Scopul principal al proiectului este realizarea unui aplicații web care să permită clasificarea de cifre desenate de către utilizatori într-o pagină web utilizând limbajul Python și framework-urile și bibliotecile puse la dispoziție de acesta. Pentru acest lucru se utilizează o rețea neuronală convoluțională antrenată local pe setul de date MNIST, o colecție de 60000 de poze cu cifre.

## Instalare
Pentru realizarea proiectului este nevoie de instalarea în prealabil a următoarelor:
- Python
- pip v23.3.2
- Pycharm sau alt IDE ce permite executarea de cod Python
- Pytorch v2.1.2
- Flask v3.0.0
- Torchvision v0.16.2

După instalarea acestora se poate clona acest repository de GIT.
Dacă se dorește se poate face reantrenarea modelului executând scriptul train.py.
În cazul în care se dorește păstrarea modelului deja antrenat, se poate porni serverul executând scriptul server.py. 
Pentru a accesa pagina Web se poate intra pe URL-ul http://127.0.0.1:5000/, dacă setările default ale flask nu au fost alterate. În caz contrar, va scrie în consolă adresa la care ascultă serverul.

## Tehnologii și librării utilizate
- Python
- Flask
- PyTorch
- CV
- Matplotlib
- Javascript
- HTML
- Chart.js

## Cum se foloseste aplicatia
Odată pornit serverul se poate naviga pe pagina Web pe care acesta ascultă și se poate desena o cifră pe canvas. 
La apăsarea butonului de "Upload" va apărea rezultatul clasificării și o distribuție de probabilități.
La apăsarea butonului de "Clear" canvas-ul va fi șters și se poate desena o nouă cifră.
