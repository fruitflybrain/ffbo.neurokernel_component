export PYTHONPATH=/neuroarch:/neurokernel:/neurodriver/:/retina:/usr/local/lib/python2.7/site-packages:/usr/lib/python2.7/dist-packages/:$PYTHONPATH

source activate NK

BASEDIR=$(dirname "$0")
cd $BASEDIR

if [ $# -eq 0 ]; then
    python neurokernel_component.py
fi

if [ $# -eq 1 ]; then
    python neurokernel_component.py --url $1
fi

if [ $# -eq 2 ]; then
  if [ $2 = "--no-ssl" ]; then
    python neurokernel_component.py --no-ssl --url $1
  else
    echo "Unrecognised argument"
  fi
fi