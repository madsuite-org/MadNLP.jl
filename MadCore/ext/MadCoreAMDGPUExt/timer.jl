MadCore.create_timer(::Type{<:ROCArray}) = MadCore.EventTimer{AMDGPU.HIP.HIPEvent}()

MadCore._new_event(::Type{AMDGPU.HIP.HIPEvent}) =
    AMDGPU.HIP.HIPEvent(AMDGPU.stream(); do_record = false, timing = true)
# A HIPEvent stores the stream it is recorded on; point it at the current one.
function MadCore._record!(e::AMDGPU.HIP.HIPEvent)
    e.stream = AMDGPU.stream().stream
    AMDGPU.HIP.record(e)
    return
end
MadCore._isdone(e::AMDGPU.HIP.HIPEvent) = AMDGPU.HIP.isdone(e)
MadCore._wait(e::AMDGPU.HIP.HIPEvent) = AMDGPU.HIP.synchronize(e)
MadCore._elapsed(start::AMDGPU.HIP.HIPEvent, stop::AMDGPU.HIP.HIPEvent) =
    Float64(AMDGPU.HIP.elapsed(start, stop))
