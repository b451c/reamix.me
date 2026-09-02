-- list top-level windows of this process that look like JUCE popups (no title or ours)
local out = {}
local cnt, list = reaper.JS_Window_ListAllTop()
for addr in string.gmatch(list or "", "[^,]+") do
  local h = reaper.JS_Window_HandleFromAddress(addr)
  if h and reaper.JS_Window_IsVisible(h) then
    local cls = reaper.JS_Window_GetClassName(h) or ""
    local title = reaper.JS_Window_GetTitle(h) or ""
    if cls:find("JUCE") or title:find("reamix") or title:find("custom") then
      local _, l, t, r, b = reaper.JS_Window_GetRect(h)
      out[#out+1] = string.format("%s|%s|%d,%d,%d,%d", cls, title, l, t, r - l, b - t)
    end
  end
end
local f = io.open("C:\\Users\\basic\\rm_out.txt", "w"); f:write(table.concat(out, "\n")); f:close()
