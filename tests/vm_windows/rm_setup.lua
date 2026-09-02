-- reamix Windows VM leg: run via `reaper.exe -nonewinst rm_setup.lua` (ReamixCmd task).
-- New project + media item selected + reamix window open; writes window rects to rm_out.txt.
local media = "C:\\Users\\basic\\Downloads\\YTDown_YouTube_Christina-Aguilera-Dirrty-Official-HD-Vi_Media_4Rg3sAb8Id8_009_128k.mp3"
local out = {}
reaper.Main_OnCommand(40023, 0)            -- new project (no prompt when clean)
reaper.InsertTrackAtIndex(0, false)
local tr = reaper.GetTrack(0, 0)
reaper.SetOnlyTrackSelected(tr)
reaper.SetEditCurPos(0, false, false)
reaper.InsertMedia(media, 0)
local it = reaper.GetTrackMediaItem(tr, 0)
if it then reaper.SetMediaItemSelected(it, true) end
reaper.UpdateArrange()
out[#out+1] = "item_len=" .. tostring(it and reaper.GetMediaItemInfo_Value(it, "D_LENGTH") or -1)
local id = reaper.NamedCommandLookup("_reamix_ShowWindow")
out[#out+1] = "cmd=" .. tostring(id)
local n = 0
local function tick()
  n = n + 1
  local st = reaper.GetToggleCommandState(id)
  if st ~= 1 and n <= 4 then reaper.Main_OnCommand(id, 0) end
  if n < 6 then reaper.defer(tick) return end
  out[#out+1] = "toggle_state=" .. tostring(reaper.GetToggleCommandState(id))
  local hw = reaper.JS_Window_Find("reamix.me", true)
  if hw then
    local _, l, t, r, b = reaper.JS_Window_GetRect(hw)
    out[#out+1] = string.format("reamix_rect=%d,%d,%d,%d", l, t, r - l, b - t)
  else
    out[#out+1] = "reamix_rect=none"
  end
  local f = io.open("C:\\Users\\basic\\rm_out.txt", "w"); f:write(table.concat(out, "\n")); f:close()
end
reaper.defer(tick)
