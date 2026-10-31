MadCore.create_timer(::Type{<:CuArray}) = MadCoreGPU.EventTimer{CuEvent}()

MadCoreGPU._new_event(::Type{CuEvent}) = CuEvent()
MadCoreGPU._record!(e::CuEvent) = record(e, stream())
MadCoreGPU._isdone(e::CuEvent) = CUDACore.isdone(e)
MadCoreGPU._wait(e::CuEvent) = CUDACore.synchronize(e)
MadCoreGPU._elapsed(start::CuEvent, stop::CuEvent) = Float64(elapsed(start, stop))
