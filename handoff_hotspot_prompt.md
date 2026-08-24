# Prompt to paste into Claude on the Mac

(Drive the Jetson hotspot setup + WLED multi-network test from this Mac.)

---

I'm setting up a Jetson Orin Nano running a project called **Collective Echo** for a gallery show with no wifi. I want the Jetson to:

- Use my **home wifi** when at home (current default)
- Use my **phone hotspot** when I enable it (away, debugging only)
- Auto-fall-back to its **own AP (CollectiveEcho-AP)** when neither is reachable (the gallery)

WLED (the LED controller) needs to follow the Jetson onto whichever network — primarily it should join CollectiveEcho-AP at the gallery and home wifi at home. Phone-hotspot scenario doesn't need lights.

I'd like you to drive the whole setup over SSH from here. WLED is configurable via its JSON API, so no browser steps required. I'll provide credentials and confirm checkpoints; otherwise be autonomous.

## State of the world (verified)

- Jetson user/host: `utku@jetson`
- Wifi card: Realtek RTL8822CE — supports AP mode but NOT concurrent client+AP (flipping to AP drops wifi SSH)
- Wifi interface name: `wlP1p1s0`
- Currently on wifi at **192.168.0.152**, gateway 192.168.0.1
- USB-C device-mode lifeline: **192.168.55.1** (l4tbr0). Test with `ssh utku@192.168.55.1 echo OK` from this Mac
- WLED controller currently at **192.168.0.11**, MAC `a4:f0:0f:91:6f:94`, firmware **WLED-MM 16.0.0** (multi-network supported), no auth
- WLED LED count: 210
- After AP is up: Jetson at **10.42.0.1**, dnsmasq DHCP at `10.42.0.10–254`
- Audio service: `collective-echo.service` (systemd, enabled). Config at `/home/utku/Desktop/projects/collectiveEcho/config.yaml`
- Power: Jetson on barrel-jack DC, USB-C is free
- Project root on Jetson: `/home/utku/Desktop/projects/collectiveEcho`

## Plan I want you to execute

Be autonomous — execute SSH commands directly, validate each step, only stop to ask me when you need a credential or hit something unexpected.

1. **Confirm USB-C lifeline.** Run `ssh utku@192.168.55.1 hostname` from this Mac. If it doesn't return `jetson`, STOP — diagnose USB-C (cable seated? Mac sees a new RNDIS/CDC-ECM ethernet adapter?) before touching anything else.

2. **Ask me for the AP SSID and password.** Suggest defaults: SSID `CollectiveEcho`, password something I approve (8+ chars, WPA2). Hold these — they'll be reused for WLED.

3. **Configure WLED via API (BEFORE touching Jetson wifi).** Over the existing wifi:
   - `GET http://192.168.0.11/json/cfg` to fetch current config
   - Add the AP SSID/password as a second client network in `nw.ins` (keep home wifi as primary so home behavior doesn't change)
   - Set the mDNS hostname under `id.mdns` to `wled` so the controller is reachable as `wled.local` regardless of subnet
   - `POST http://192.168.0.11/json/cfg` with the modified config
   - WLED reboots in ~5s. Wait, then verify `curl http://192.168.0.11/json/info` still returns 200 and the new hostname is set
   - Test mDNS works from Jetson: `ssh utku@192.168.0.152 "ping -c 2 wled.local"` — should resolve

4. **Create the AP NetworkManager connection on Jetson with `autoconnect no`** for now (so a reboot reverts to current behavior until we've verified):
   ```bash
   sudo nmcli connection add type wifi ifname wlP1p1s0 \
       con-name CollectiveEcho-AP ssid "<SSID>"
   sudo nmcli connection modify CollectiveEcho-AP \
       802-11-wireless.mode ap \
       802-11-wireless.band bg \
       ipv4.method shared \
       wifi-sec.key-mgmt wpa-psk \
       wifi-sec.psk "<PASSWORD>" \
       connection.autoconnect no
   ```

5. **Test the AP fallback path.** This drops wifi SSH — switch to USB-C for everything that follows.
   - `sudo nmcli connection up CollectiveEcho-AP` — wifi SSH dies
   - From here on, SSH via `ssh utku@192.168.55.1`
   - Wait 30s for WLED to fail over from home wifi to the AP
   - `sudo cat /var/lib/NetworkManager/dnsmasq-shared.leases` — should show a lease for MAC `a4:f0:0f:91:6f:94` at some `10.42.0.X`
   - `curl -s http://wled.local/json/info | python3 -m json.tool | head -20` — confirm WLED is reachable via mDNS on the AP subnet
   - If both checks pass, the gallery scenario is proven

6. **Restore client mode.** While still on USB-C SSH:
   ```bash
   sudo nmcli connection down CollectiveEcho-AP
   sudo nmcli connection up "<HomeWifiName>"   # find name with: nmcli -t -f NAME,TYPE c
   ```
   - Wait 30s for WLED to fall back to home wifi
   - `curl -s http://wled.local/json/info` should resolve to `192.168.0.11` again

7. **Promote AP to auto-fallback.** Now that we know everything works, set the priority chain so it auto-picks the right network at boot:
   ```bash
   sudo nmcli connection modify "<HomeWifiName>" connection.autoconnect yes connection.autoconnect-priority 200
   sudo nmcli connection modify CollectiveEcho-AP connection.autoconnect yes connection.autoconnect-priority 50
   ```
   At boot the Jetson tries home wifi first; if SSID is in range it joins; if not, AP comes up.

8. **Optionally save my phone hotspot** as a middle priority. Ask me for phone hotspot SSID + password (only if I want this — I might skip).
   ```bash
   sudo nmcli device wifi connect "<phone-ssid>" password "<phone-pass>"   # save+join
   sudo nmcli connection modify "<phone-ssid>" connection.autoconnect yes connection.autoconnect-priority 100
   sudo nmcli connection up "<HomeWifiName>"   # back to home for now
   ```

9. **Update `config.yaml`** to use mDNS instead of hard-coded WLED IP (so the same config works on home wifi or AP):
   - `wled.host: "http://wled.local"` (was `http://192.168.0.11`)
   - `wled.realtime.host: "wled.local"` (was `192.168.0.11`)

   ```bash
   sudo systemctl restart collective-echo
   journalctl -u collective-echo -f
   ```

10. **End-to-end test.** I'll trigger "i love you" near the Jetson's mic. You watch the journal and confirm:
    - `Vosk detected 'i love you' [ily]`
    - `Response (early): LAYERED N clips`
    - `WLED preset N activated` AND `WLEDRealtime → wled.local:21324, 210 LEDs` (proves both HTTP and UDP paths reach WLED)

11. **Boot persistence sanity check.** Suggest I run `sudo reboot` and confirm the Jetson comes back up on home wifi automatically (at home), with the AP ready to take over if home wifi disappears.

## What NOT to do

- Don't `sudo nmcli connection delete` anything — only modify autoconnect/priority
- Don't enable AP autoconnect (step 7) until step 5 + 6 are both verified
- Don't reboot the Jetson before USB-C SSH is confirmed working
- Don't change `config.yaml` outside the two `wled.host`/`wled.realtime.host` lines — every other knob has been tuned for the show

## Failure modes to flag and stop on

- Step 3 WLED API errors (4xx/5xx) — stop, report, don't touch Jetson wifi
- Step 5 dnsmasq leases empty after 60s — WLED didn't join the AP. Stop, report. I might need to power-cycle WLED.
- Step 6 WLED unreachable on home wifi after 60s — WLED's wifi config is broken. Worst case I flash it. Stop and tell me.
- Step 10 WLED not reached even though `wled.local` resolves — likely a config typo. Show me the relevant `config.yaml` lines.

Start by running the USB-C lifeline check and asking me for the AP SSID + password.
