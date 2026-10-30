MadNLP.create_timer(::Type{<:CuArray}) = MadNLPGPU.EventTimer{CuEvent}()

MadNLPGPU._new_event(::Type{CuEvent}) = CuEvent()
MadNLPGPU._record!(e::CuEvent) = record(e, stream())
MadNLPGPU._isdone(e::CuEvent) = CUDACore.isdone(e)
MadNLPGPU._wait(e::CuEvent) = CUDACore.synchronize(e)
MadNLPGPU._elapsed(start::CuEvent, stop::CuEvent) = Float64(elapsed(start, stop))
