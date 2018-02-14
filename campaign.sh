#!/usr/bin/env bash
rm log.txt;
mkdir results;
export EXP_INTERP='/usr/bin/python' ;

echo '=================> Performing RPE';
echo '=================> Rpe Armball';
EXP_DATE=$(date +"%s")
$EXP_INTERP rpe.py armball --path=results --name='Rpe Armball '$EXP_DATE || echo 'FAILURE';
echo '=================> Rpe Armarrow';
EXP_DATE=$(date +"%s")
$EXP_INTERP rpe.py armarrow --path=results --name='Rpe Armarrow '$EXP_DATE || echo 'FAILURE';

echo '=================> Performing RGE-EFR';
echo '=================> Rge-Efr Armball';
EXP_DATE=$(date +"%s")
$EXP_INTERP rge_efr.py armball --path=results --name='Rge-Efr Armball '$EXP_DATE || echo 'FAILURE';
echo '=================> Rge-Efr Armarrow';
EXP_DATE=$(date +"%s")
$EXP_INTERP rge_efr.py armarrow --path=results --name='Rge-Efr Armarrow '$EXP_DATE || echo 'FAILURE';

echo '=================> Performing RGE-REP 10 Latents Kde';
echo '=================> Rge-Pca Armball';
EXP_DATE=$(date +"%s")
$EXP_INTERP rge_rep.py pca armball --sampling=kde --nlatents=10 --path=results --name='Rge-Pca Armball 10L Kde '$EXP_DATE || echo 'FAILURE';
echo '=================> Rge-Pca Armarrow';
EXP_DATE=$(date +"%s")
$EXP_INTERP rge_rep.py pca armarrow --sampling=kde --nlatents=10 --path=results --name='Rge-Pca Armarrow 10L Kde '$EXP_DATE || echo 'FAILURE';
echo '=================> Rge-Ae Armball';
EXP_DATE=$(date +"%s")
$EXP_INTERP rge_rep.py ae armball --sampling=kde --nlatents=10 --path=results --name='Rge-Ae Armball 10L Kde '$EXP_DATE || echo 'FAILURE';
echo '=================> Rge-Ae Armarrow';
EXP_DATE=$(date +"%s")
$EXP_INTERP rge_rep.py ae armarrow --sampling=kde --nlatents=10 --path=results --name='Rge-Ae Armarrow 10L Kde '$EXP_DATE || echo 'FAILURE';
echo '=================> Rge-Vae Armball';
EXP_DATE=$(date +"%s")
$EXP_INTERP rge_rep.py vae armball --sampling=kde --nlatents=10 --path=results --name='Rge-Vae Armball 10L Kde '$EXP_DATE || echo 'FAILURE';
echo '=================> Rge-Vae Armarrow';
EXP_DATE=$(date +"%s")
$EXP_INTERP rge_rep.py vae armarrow --sampling=kde --nlatents=10 --path=results --name='Rge-Vae Armarrow 10L Kde '$EXP_DATE || echo 'FAILURE';
echo '=================> Rge-Rfvae Armball';
EXP_DATE=$(date +"%s")
$EXP_INTERP rge_rep.py rfvae armball --sampling=kde --nlatents=10 --path=results --name='Rge-Rfvae Armball 10L Kde '$EXP_DATE || echo 'FAILURE';
echo '=================> Rge-Rfvae Armarrow';
EXP_DATE=$(date +"%s")
$EXP_INTERP rge_rep.py rfvae armarrow --sampling=kde --nlatents=10 --path=results --name='Rge-Rfvae Armarrow 10L Kde '$EXP_DATE || echo 'FAILURE';
echo '=================> Rge-Isomap Armball';
EXP_DATE=$(date +"%s")
$EXP_INTERP rge_rep.py isomap armball --sampling=kde --nlatents=10 --path=results --name='Rge-Isomap Armball 10L Kde '$EXP_DATE || echo 'FAILURE';
echo '=================> Rge-Isomap Armarrow';
EXP_DATE=$(date +"%s")
$EXP_INTERP rge_rep.py isomap armarrow --path=results --name='Rge-Isomap Armarrow 10L Kde '$EXP_DATE || echo 'FAILURE';

echo '=================> Performing RGE-REP 3/2 Latents Kde';
echo '=================> Rge-Pca Armball';
EXP_DATE=$(date +"%s")
$EXP_INTERP rge_rep.py pca armball --sampling=kde --nlatents=2 --path=results --name='Rge-Pca Armball 2L Kde '$EXP_DATE || echo 'FAILURE';
echo '=================> Rge-Pca Armarrow';
EXP_DATE=$(date +"%s")
$EXP_INTERP rge_rep.py pca armarrow --sampling=kde --nlatents=3 --path=results --name='Rge-Pca Armarrow 3L Kde '$EXP_DATE || echo 'FAILURE';
echo '=================> Rge-Ae Armball';
EXP_DATE=$(date +"%s")
$EXP_INTERP rge_rep.py ae armball --sampling=kde --nlatents=2 --path=results --name='Rge-Ae Armball 2L Kde '$EXP_DATE || echo 'FAILURE';
echo '=================> Rge-Ae Armarrow';
EXP_DATE=$(date +"%s")
$EXP_INTERP rge_rep.py ae armarrow --sampling=kde --nlatents=3 --path=results --name='Rge-Ae Armarrow 3L Kde '$EXP_DATE || echo 'FAILURE';
echo '=================> Rge-Vae Armball';
EXP_DATE=$(date +"%s")
$EXP_INTERP rge_rep.py vae armball --sampling=kde --nlatents=2 --path=results --name='Rge-Vae Armball 2L Kde '$EXP_DATE || echo 'FAILURE';
echo '=================> Rge-Vae Armarrow';
EXP_DATE=$(date +"%s")
$EXP_INTERP rge_rep.py vae armarrow --sampling=kde --nlatents=3 --path=results --name='Rge-Vae Armarrow 3L Kde '$EXP_DATE || echo 'FAILURE';
echo '=================> Rge-Rfvae Armball';
EXP_DATE=$(date +"%s")
$EXP_INTERP rge_rep.py rfvae armball --sampling=kde --nlatents=2 --path=results --name='Rge-Rfvae Armball 2L Kde '$EXP_DATE || echo 'FAILURE';
echo '=================> Rge-Rfvae Armarrow';
EXP_DATE=$(date +"%s")
$EXP_INTERP rge_rep.py rfvae armarrow --sampling=kde --nlatents=3 --path=results --name='Rge-Rfvae Armarrow 3L Kde '$EXP_DATE || echo 'FAILURE';
echo '=================> Rge-Isomap Armball';
EXP_DATE=$(date +"%s")
$EXP_INTERP rge_rep.py isomap armball --sampling=kde --nlatents=2 --path=results --name='Rge-Isomap Armball 2L Kde '$EXP_DATE || echo 'FAILURE';
echo '=================> Rge-Isomap Armarrow';
EXP_DATE=$(date +"%s")
$EXP_INTERP rge_rep.py isomap armarrow --sampling=kde --nlatents=3 --path=results --name='Rge-Isomap Armarrow 3L Kde '$EXP_DATE || echo 'FAILURE';


echo '=================> Performing RGE-VAE 10 Latents Normal';
echo '=================> Rge-Vae Armball';
EXP_DATE=$(date +"%s")
$EXP_INTERP rge_rep.py vae armball --sampling=normal --nlatents=10 --path=results --name='Rge-Vae Armball 10L Normal '$EXP_DATE || echo 'FAILURE';
echo '=================> Rge-Vae Armarrow';
EXP_DATE=$(date +"%s")
$EXP_INTERP rge_rep.py vae armarrow --sampling=normal --nlatents=10 --path=results --name='Rge-Vae Armarrow 10L Normal $EXP_DATE' || echo 'FAILURE';