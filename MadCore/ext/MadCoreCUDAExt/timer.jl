MadCore.create_timer(::Type{<:CuArray}) = MadCore.EventTimer{CuEvent}()

MadCore._new_event(::Type{CuEvent}) = CuEvent()
MadCore._record!(e::CuEvent) = record(e, stream())
MadCore._isdone(e::CuEvent) = CUDACore.isdone(e)
MadCore._wait(e::CuEvent) = CUDACore.synchronize(e)
MadCore._elapsed(start::CuEvent, stop::CuEvent) = Float64(elapsed(start, stop))
