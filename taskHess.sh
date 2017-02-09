echo python $1 -R 10 -size 5000 -A 8.0 -B 7.0 -o $2 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 5000 -A 2.0 -B 10.0 -o $2 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 5000 -A 0.6 -B 0.9 -o $2 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 5000 -A 3.0 -B 0.8 -o $2 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 5000 -A 4.0 -B 3.0 -o $2 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 15000 -A 8.0 -B 7.0 -o $2 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 15000 -A 2.0 -B 10.0 -o $2 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 15000 -A 0.6 -B 0.9 -o $2 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 15000 -A 3.0 -B 0.8 -o $2 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 15000 -A 4.0 -B 3.0 -o $2 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 20000 -A 8.0 -B 7.0 -o $2 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 20000 -A 2.0 -B 10.0 -o $2 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 20000 -A 0.6 -B 0.9 -o $2 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 20000 -A 3.0 -B 0.8 -o $2 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 20000 -A 4.0 -B 3.0 -o $2 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 5000 -A 8.0 -B 7.0 -o $2 -isNoise -std 0.1 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 5000 -A 8.0 -B 7.0 -o $2 -isNoise -std 0.2 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 5000 -A 2.0 -B 10.0 -o $2 -isNoise -std 0.1 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 5000 -A 2.0 -B 10.0 -o $2 -isNoise -std 0.2 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 5000 -A 0.6 -B 0.9 -o $2 -isNoise -std 0.1 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 5000 -A 0.6 -B 0.9 -o $2 -isNoise -std 0.2 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 5000 -A 3.0 -B 0.8 -o $2 -isNoise -std 0.1 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 5000 -A 3.0 -B 0.8 -o $2 -isNoise -std 0.2 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 5000 -A 4.0 -B 3.0 -o $2 -isNoise -std 0.1 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 5000 -A 4.0 -B 3.0 -o $2 -isNoise -std 0.2 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 15000 -A 8.0 -B 7.0 -o $2 -isNoise -std 0.1 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 15000 -A 8.0 -B 7.0 -o $2 -isNoise -std 0.2 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 15000 -A 2.0 -B 10.0 -o $2 -isNoise -std 0.1 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 15000 -A 2.0 -B 10.0 -o $2 -isNoise -std 0.2 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 15000 -A 0.6 -B 0.9 -o $2 -isNoise -std 0.1 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 15000 -A 0.6 -B 0.9 -o $2 -isNoise -std 0.2 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 15000 -A 3.0 -B 0.8 -o $2 -isNoise -std 0.1 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 15000 -A 3.0 -B 0.8 -o $2 -isNoise -std 0.2 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 15000 -A 4.0 -B 3.0 -o $2 -isNoise -std 0.1 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 15000 -A 4.0 -B 3.0 -o $2 -isNoise -std 0.2 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 20000 -A 8.0 -B 7.0 -o $2 -isNoise -std 0.1 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 20000 -A 8.0 -B 7.0 -o $2 -isNoise -std 0.2 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 20000 -A 2.0 -B 10.0 -o $2 -isNoise -std 0.1 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 20000 -A 2.0 -B 10.0 -o $2 -isNoise -std 0.2 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 20000 -A 0.6 -B 0.9 -o $2 -isNoise -std 0.1 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 20000 -A 0.6 -B 0.9 -o $2 -isNoise -std 0.2 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 20000 -A 3.0 -B 0.8 -o $2 -isNoise -std 0.1 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 20000 -A 3.0 -B 0.8 -o $2 -isNoise -std 0.2 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 20000 -A 4.0 -B 3.0 -o $2 -isNoise -std 0.1 -Hess | submitAll.pl -l walltime=03:59:00
echo python $1 -R 10 -size 20000 -A 4.0 -B 3.0 -o $2 -isNoise -std 0.2 -Hess | submitAll.pl -l walltime=03:59:00