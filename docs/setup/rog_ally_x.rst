Go the device manager, find the COM port, go to properties, advanced, and change the latency timer value.

Step 1: Enable Windows NTP Server in the Registry
Open Registry Editor:

Press Win + R, type regedit, and hit Enter.
Navigate to:

HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Services\W32Time\Config
Modify AnnounceFlags:

Double-click AnnounceFlags.
Change the value to 5 (to enable NTP server mode).
Click OK.
Enable NTP Server: Navigate to:

HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Services\W32Time\TimeProviders\NtpServer
Double-click Enabled.
Set its value to 1.
Click OK.
Step 2: Configure Windows Time Service
Open Command Prompt as Administrator.

Enable Windows Time Service:

sc config w32time start=auto
net start w32time
Set Windows as an NTP Server:

w32tm /config /manualpeerlist:"time.windows.com" /syncfromflags:manual /update
Force Windows to Act as an NTP Server:



w32tm /config /reliable:YES
w32tm /resync
Step 3: Check NTP Server Status
Verify Service Running:



net start w32time
Check NTP Server Configuration:



w32tm /query /configuration
Check Current Time Status:



w32tm /query /status
Verify NTP Server Peers:



w32tm /query /Peers

Step 5: Test NTP Server from Ubuntu
On Ubuntu, run:



sudo ntpdate <windows_ip_address>
