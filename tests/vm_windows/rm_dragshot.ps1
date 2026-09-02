# drag (coords in rm_drag.txt "x1 y1 x2 y2"), wait, then screenshot in the SAME task so the
# JUCE CallOutBox (modal, dismissed on focus loss) is still up when the pixels are read.
$c = (Get-Content C:\Users\basic\rm_drag.txt -Raw).Trim().Split(' ')
$x1=[int]$c[0]; $y1=[int]$c[1]; $x2=[int]$c[2]; $y2=[int]$c[3]
Add-Type @"
using System; using System.Runtime.InteropServices;
public class MD {
  [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
  [DllImport("user32.dll")] public static extern void mouse_event(uint f, uint dx, uint dy, uint d, UIntPtr e);
  [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
}
"@
[MD]::SetProcessDPIAware() | Out-Null
[MD]::SetCursorPos($x1, $y1) | Out-Null; Start-Sleep -Milliseconds 150
[MD]::mouse_event(0x0002, 0, 0, 0, [UIntPtr]::Zero); Start-Sleep -Milliseconds 120
for ($i = 1; $i -le 12; $i++) {
  [MD]::SetCursorPos([int]($x1 + ($x2 - $x1) * $i / 12), [int]($y1 + ($y2 - $y1) * $i / 12)) | Out-Null
  Start-Sleep -Milliseconds 30
}
Start-Sleep -Milliseconds 120
[MD]::mouse_event(0x0004, 0, 0, 0, [UIntPtr]::Zero)
Start-Sleep -Milliseconds 1500
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$b = [System.Windows.Forms.SystemInformation]::VirtualScreen
$bmp = New-Object System.Drawing.Bitmap $b.Width, $b.Height
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen($b.Left, $b.Top, 0, 0, $bmp.Size)
$bmp.Save("C:\Users\basic\rm_dragshot.png")
$g.Dispose(); $bmp.Dispose()
"done" | Out-File C:\Users\basic\rm_drag_done.txt -Encoding ascii
