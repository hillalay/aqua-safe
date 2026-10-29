# Kişi takibi ve işlenmiş MP4

Site veya tarayıcı gerekmez. Her tespit edilen kişi ayrı kimlikle takip edilir.
16 gerçek örnek bulunan dizilerde kutuda NORMAL YÜZÜCÜ veya BOĞULMA DAVRANIŞI
ve yüzde boğulma skoru gösterilir. Daha kısa dizilerde YETERSİZ VERİ yazılır; skor üretilmez.
Normal etiketinde de yüzde aynı boğulma sınıfının skorudur; %50 gösterim eşiğidir.

## Kullanım

PowerShell, proje klasöründe:

```powershell
.\.venv\Scripts\python.exe -m src.drowning_demo.swimmers "video.mp4"
```

Birden fazla video yolu eklenebilir. CUDA varsa otomatik kullanılır;
CPU için `--device cpu` ekleyin. Çıktının tam yolu işlem sonunda yazdırılır:
`results/drowning_demo/<benzersiz-klasör>/annotated_video.mp4`.
MP4 dosyasını bilgisayarınızdaki video oynatıcıyla açabilirsiniz.
JSON dosyası kişi kimliklerini, kutuları ve zaman penceresi skorlarını içerir.
Bütün kareler, çözünürlük ve sunum zamanları doğrulanır. MP4 sessizdir.
Her video için ayrı takipçi kurulur. Eksik tespitte skor dizisi bölünür.

`video_isle.cmd` aynı parametreleri destekler:

```powershell
.\video_isle.cmd "video.mp4" --checkpoint "results/swimmer_training/mil/fold_0/best_checkpoint.pt"
```

## Kurulum

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
```

Yerel `yolov8s.pt` kişi tespiti ve ByteTrack takibi için kullanılır.
Takip API'si: https://docs.ultralytics.com/modes/track/

## Eğitim

Normal yüzme stilleri tek `normal_swimming` sınıfında birleştirilir;
diğer sınıf `drowning_behavior` olur. Video düzeyindeki zayıf etiketlerden
MIL eğitimi yapılır. İlişkili videolar aynı fold'da tutulur.
Yeniden eğitim ayrı klasöre yazılır:

```powershell
.\.venv\Scripts\python.exe -m src.video_classification.mil_train --config results/swimmer_training/config.yaml --fold 0
```

Yeniden eğitilmiş ağırlığı kullanmak için:

```powershell
.\.venv\Scripts\python.exe -m src.drowning_demo.swimmers "video.mp4" --checkpoint results/swimmer_training/mil/fold_0/best_checkpoint.pt
```

## Sınırlar

Yüzde, modelin davranış skorudur; kalibre edilmiş gerçek boğulma olasılığı değildir.
Video etiketleri kişi veya olay zamanı etiketleri sağlamaz. Kişi kırpımları
üzerindeki başarı ayrıca doğrulanmamıştır; mevcut doğrulamada yanlış alarmlar yüksektir.
YOLO havuz kenarındaki insanları da tespit edebilir; su altındaki kişiler kaçabilir,
örtüşmede takip kimliği değişebilir. Kısa dizilerde kare tekrarı yapılmaz ve
davranış sınıfı/skoru verilmez; gerçek örnek sayısı JSON'da saklanır.
Tespit olmayan karelerde KİŞİ TESPİT EDİLEMEDİ yazılır.
Çevrimdışı en yakın pencere skoru gösterilir.
Can güvenliği kararı vermek için kullanılmamalıdır.

## Kişi bazlı veri hazırlama pilotu

Yalnızca `kelebek2.mp4`, `bogulma1.mp4` ve `data/test_videos/1020(3)-1.mp4`
işlenir. MIL modeli yüklenmez ve eğitim yapılmaz:

```powershell
.\.venv\Scripts\python.exe -m src.person_dataset prepare
.\.venv\Scripts\python.exe -m src.person_dataset validate --csv results/person_dataset_pilot/clips.csv
```

Hazırlama mevcut çıktı klasörünün üzerine yazmaz. Yeniden çalıştırmak gerekirse
`prepare --output results/person_dataset_pilot_2` ile yeni bir klasör seçin.

`results/person_dataset_pilot/clips.csv` dosyasını Excel veya bir CSV düzenleyicide
açın. `clip_path` MP4'ü, `preview_path` aynı klibin farklı gerçek karelerini gösterir.
`contact_sheet.jpg` tüm adayları listeler; numaralar CSV satır sırasına karşılık gelir.
MP4'ler sessiz, 320×320 ve özgün kare zamanlarıyla kaydedilir. Kare tekrarı veya süre
uzatma yapılmaz. Başka kişi sabit kırpıma giriyorsa MP4 dışa aktarılmaz; CSV'de
gerekçe ve önizleme korunur. Kısa MP4'ler yalnızca inceleme içindir.

Yalnızca şu sütunları düzenleyin:

- `label`: `NORMAL`, `DROWNING` veya `UNCERTAIN`.
- `review_status`: `unreviewed`, `verified` veya `excluded`.
- `notes`: inceleme notunuz.
- `split`: başlangıçta `unassigned`; diğer değerler `train`, `validation`, `test`.

`verified`, klibin baştan sona aynı kişiyi içerdiğini ve davranış etiketinin doğru
olduğunu izleyerek doğruladığınız anlamına gelir. Emin değilseniz `UNCERTAIN` bırakın.
Klasör adından etiket atanmaz. Normal yüzme stilleri aynı `NORMAL` sınıfına girer.

Dosyayı UTF-8 CSV olarak kaydedip doğrulama komutunu çalıştırın. Etiketlerinizi
`clips.csv` içinde korur; hesaplanmış `training_eligible` alanını
`validated_clips.csv` dosyasına yazar. Eğitim girdisi olarak kullanılabilecek
kontrollü çıktı budur; doğrulama komutu eğitim başlatmaz.
Teknik geçersizlik, yetersiz gerçek kare, `UNCERTAIN` etiketi veya doğrulanmamış
inceleme durumunda `training_eligible=False` kalır. Korunan sütunları değiştirmeyin.

`pilot_report.json` kişi/ID sayısını, kopmaları, şüpheleri ve geçersiz örnekleri;
`*_detections.jsonl` tespit olmayanlar dahil her kaynak kareyi kaydeder.
`manifest.json` kaynakları, koordinatları ve dosya hash'lerini korur.
Kaynak/çekim bilgileri bilinmediğinden pilot tek `UNRESOLVED_CAPTURE_PILOT`
çekim grubunda tutulur. Bu grup farklı kümelere dağıtılırsa doğrulama reddedilir.
Bağımsız test ayrımı için önce çekim kaynaklarının ayrıca doğrulanması gerekir.

## Yeni videoların eklenmesi

Yeni envanter ve çıktılar önceki pilotu değiştirmeden
`results/person_dataset_new_videos/` altında tutulur:

```powershell
.\.venv\Scripts\python.exe -m src.person_dataset_update inventory
.\.venv\Scripts\python.exe -m src.person_dataset_update compare
```

`inventory.csv` bütün `data/` video dosyalarını tarar; eski kopyaları yeni kaynak
olarak saymaz. Yeni benzersiz kaynaklar tamamen çözülür. `inventory_report.json`
sayımın kapsamını, `source_candidates.csv` aynı çekimden gelme adaylarını gösterir.
Klasör kaynak kategorisi yalnızca ipucudur; klip etiketi değildir.

`tracker_comparison/` içindeki kontakt sayfalarını insan tespiti ve ID sürekliliği
açısından inceleyin. Karşılaştırma raporunun `selection` alanındaki `status=reviewed`
ve `tracker` seçimi görsel inceleme sonucunu kaydeder; klip sayısı tek başına seçim
ölçütü değildir. İnceleme kaydı olmadan yeni veri hazırlama başlamaz:

```powershell
.\.venv\Scripts\python.exe -m src.person_dataset_update prepare-new
.\.venv\Scripts\python.exe -m src.person_dataset validate --csv results/person_dataset_new_videos/dataset/clips.csv
```

Yeni CSV, kontakt sayfası, kişi MP4'leri ve raporlar `dataset/` altındadır.
Bütün etiketler `UNCERTAIN` başlar. Etiketleme ve doğrulama kuralları önceki
pilotla aynıdır. Minimum uzunluk sabit 2 saniye değil, mevcut R3D-18 örneklemesinde
16 gerçek karedir: `frame_step=max(1, round(source_fps/8))`, minimum kaynak kare
sayısı `15*frame_step+1`. Yaklaşık 2,5 saniyelik parçalar en fazla 3 saniyeye
tamamlanabilir veya son iki parça dengelenebilir; kişi veya kopuk diziler birleşmez.
Kare tekrarı yapılmaz. Mevcut pilot CSV ve doğrulama dosyaları güncellenmez.

Hazırlama kesilirse `prepare-new --resume` kayıtlı tespit kararlarını ve mevcut
MP4 kare/zaman bilgilerini doğrulayarak kalan kaynakları tamamlar. Tamamlanmış
CSV veya manifest bulunan veri setini tekrar işlemeyi reddeder; incelenmiş
etiketler ve önceki çıktılar üzerine yazılmaz.
