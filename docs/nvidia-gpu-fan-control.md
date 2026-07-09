# NVIDIA GPU Fan Control

Manual fan control is done with `nvidia-settings`, not `nvidia-smi`.
`nvidia-smi` can report fan speed, but it generally cannot set fan speed.

On headless hosts, `nvidia-settings` needs an X/NV-CONTROL display. On
`192.168.0.196`, a temporary Xorg control display was started on `:99`.

## Set fans to 80%

```bash
DISPLAY=:99 sudo nvidia-settings -a "[gpu:0]/GPUFanControlState=1"
DISPLAY=:99 sudo nvidia-settings -a "[fan:0]/GPUTargetFanSpeed=80"
DISPLAY=:99 sudo nvidia-settings -a "[fan:1]/GPUTargetFanSpeed=80"
```

## Verify fan target and current speed

```bash
DISPLAY=:99 sudo nvidia-settings \
  -q "[fan:0]/GPUTargetFanSpeed" \
  -q "[fan:1]/GPUTargetFanSpeed"

nvidia-smi --query-gpu=index,fan.speed,temperature.gpu,power.draw,power.limit \
  --format=csv,noheader,nounits
```

## Return to automatic fan control

```bash
DISPLAY=:99 sudo nvidia-settings -a "[gpu:0]/GPUFanControlState=0"
```

## Temporary Xorg control session

The current temporary Xorg control session on `192.168.0.196` was started with
a generated config in `/tmp/xorg-nvidia-fan.conf`.

Useful paths:

```text
/tmp/xorg-nvidia-fan.conf
/tmp/Xorg-fan.log
/tmp/Xorg-fan.pid
```

Check whether it is still running:

```bash
cat /tmp/Xorg-fan.pid
ps -o pid,stat,etime,cmd -p "$(cat /tmp/Xorg-fan.pid)"
```

If the display is not running, recreate it:

```bash
sudo nvidia-xconfig \
  --allow-empty-initial-configuration \
  --enable-all-gpus \
  --cool-bits=4 \
  --busid=PCI:33:0:0 \
  -o /tmp/xorg-nvidia-fan.conf

nohup sudo Xorg :99 \
  -config /tmp/xorg-nvidia-fan.conf \
  -noreset \
  +extension GLX \
  +extension RANDR \
  +extension RENDER \
  -logfile /tmp/Xorg-fan.log \
  >/tmp/Xorg-fan.out 2>&1 &

echo "$!" | sudo tee /tmp/Xorg-fan.pid
```
