"""New 06D harmless executable fixtures; no 06C frozen asset is changed or executed."""

import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

from packages.grounding.models import digest, seal
from packages.grounding.synthetic import ref
from packages.integrations.medgraph.resources import LocalResourcePolicy
from packages.object_store import checksum
from packages.tool_admission import AdmissionSelection, InvocationInput, ToolAdmissionService
from packages.tool_admission.models import AdmissionPolicy
from packages.tool_execution import (
    ExecutionPolicy,
    PNGPolicy,
    TechnicalQCRegistry,
    ToolExecutionService,
)
from packages.tool_workflow import CommandAllowlist, FormalContext, ToolWorkflowCompiler
from packages.tool_workflow.models import OutputSpec
from tests.tool_admission.support import fixture, reseal
from tests.tool_workflow.support import Allowlists, choose, config

FIXTURE = """import os, sys, time, struct, zlib
mode = sys.argv[-1]
inputs = [p for p in sys.argv[1:-1] if p.startswith("/in/")]
output = next(p for p in sys.argv[1:-1] if p.startswith("/out/"))
if mode == "nonzero": sys.exit(7)
if mode == "timeout": time.sleep(30)
if mode == "cpu":
 while True: pass
if mode == "memory":
 try: x = bytearray(1024**3)
 except MemoryError: sys.exit(42)
if mode == "fork":
 pid = os.fork()
 if pid == 0:
  time.sleep(30)
  open(output + ".escaped", "wb").write(b"bad")
  os._exit(0)
 print(pid, flush=True)
if mode == "cancel":
 print("ready", flush=True)
 time.sleep(30)
if mode == "logs":
 while True: print("x"*8192, flush=True)
if mode == "missing": sys.exit(0)
if mode == "oversize":
 import signal
 signal.signal(signal.SIGXFSZ,signal.SIG_DFL)
 open(output,"wb").write(b"x"*(5*1024*1024))
 sys.exit(0)
if mode == "wrong":
 open(output,"wb").write(b"not a png")
 sys.exit(0)
if mode == "symlink":
 os.symlink(inputs[0],output)
 sys.exit(0)
if mode == "isolation":
 import socket
 assert not any(k in os.environ for k in ("DATABASE_URL","API_KEY","PGPASSWORD","PYTHONPATH"))
 assert os.geteuid() != 0
 assert open(inputs[0],"rb").read(8)==b"\\x89PNG\\r\\n\\x1a\\n"
 try: socket.socket()
 except PermissionError: pass
 else: raise AssertionError("network allowed")
 try: open("/etc/passwd", "rb")
 except FileNotFoundError: pass
 else: raise AssertionError("host filesystem visible")
 try: open(inputs[0], "wb")
 except PermissionError: pass
 else: raise AssertionError("input writable")
 try: os.setsid()
 except PermissionError: pass
 else: raise AssertionError("process escaped group")
 assert len(os.sched_getaffinity(0)) == 1
h,w = 2,3
# This fixture actually consumes every supplied input. Its own PNG subset has
# one IDAT chunk and filter 0. Upstream zeros become ones; consuming that new
# intermediate output changes the next result back to zeros.
input_values = [zlib.decompress(open(p,"rb").read()[41:-16])[1] for p in inputs]
value = 9 if mode == "badlabels" else (max(input_values)+1)%2
if mode == "shape": w=4
chunk=lambda k,b: struct.pack(">I",len(b))+k+b+struct.pack(">I",zlib.crc32(k+b))
png=b"\\x89PNG\\r\\n\\x1a\\n"+chunk(b"IHDR",struct.pack(">IIBBBBB",w,h,8,0,0,0,0))
png+=chunk(b"IDAT",zlib.compress((b"\\0"+bytes([value])*w)*h))+chunk(b"IEND",b"")
with open(output,"xb") as stream: stream.write(png)
"""


def runtime_files(root):
    """Freeze an explicit stdlib-only test runtime closure, including linked libraries."""
    interpreter = Path(sys.executable).resolve()
    prefix = interpreter.parent.parent
    stdlib = prefix / "lib/python3.12"
    archive = root / "stdlib.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED) as z:
        for path in sorted(stdlib.rglob("*.py")):
            if not {"site-packages", "__pycache__", "test", "tests", "idlelib", "tkinter"} & set(
                path.relative_to(stdlib).parts
            ):
                z.write(path, str(path.relative_to(stdlib)))
    sources = [
        (interpreter, str(interpreter), True),
        (archive, str(prefix / "lib/python312.zip"), False),
        (stdlib / "os.py", str(stdlib / "os.py"), False),
    ]
    for name in ("_struct", "zlib", "_socket", "select", "array", "math"):
        source = next((stdlib / "lib-dynload").glob(name + ".*.so"))
        sources.append((source, str(source), False))
    for name in (
        "libpthread.so.0",
        "libdl.so.2",
        "libutil.so.1",
        "libm.so.6",
        "libc.so.6",
        "libz.so.1",
    ):
        target = Path("/lib/x86_64-linux-gnu") / name
        sources.append((target.resolve(), str(target), True))
    target = Path("/lib64/ld-linux-x86-64.so.2")
    sources.append((target.resolve(), str(target), True))
    # Sources may be symlinks in the installation. Resolve at deployment assembly,
    # freeze exact target bytes; execution never resolves a new symlink.
    return tuple(
        dict(source=str(s.resolve()), target=t, checksum=checksum(s.read_bytes()), executable=x)
        for s, t, x in sources
    )


def prepared(
    root,
    *,
    mode="success",
    chain=False,
    repository=None,
    base=None,
    files=None,
    output_artifact_type=None,
    mode_by_implementation=None,
):
    root.mkdir(parents=True, exist_ok=True)
    f = base or fixture(root, chain=chain)
    cfg = config()
    cfg = reseal(cfg, memory_bytes=268435456)
    executable = Path(sys.executable).resolve()
    script = root / "harmless_06d.py"
    script.write_text(FIXTURE)
    reader, invocations, policies = Allowlists(), [], {}
    for old in list(f.owners.manifests.values()):
        profile = f.owners.profiles[old.checksum]
        old_probe = f.owners.probes[profile.checksum]
        runtime = {
            **profile.runtime.model_dump(mode="json"),
            "interpreter_uri": executable.as_uri(),
            "interpreter_checksum": checksum(executable.read_bytes()),
        }
        resource = dict(
            resource_key="fixture",
            resource_role="source_code",
            uri=script.as_uri(),
            sha256=checksum(script.read_bytes())[7:],
            byte_size=script.stat().st_size,
            required=True,
            metadata={"synthetic": True},
        )
        argv = [str(executable), str(script)]
        argv += ["{input:" + p.port_name + "}" for p in old.ports if p.direction == "input"]
        argv += ["{output:" + p.port_name + "}" for p in old.ports if p.direction == "output"]
        argv += [(mode_by_implementation or {}).get(old.implementation_ref.code, mode)]
        invocation = dict(
            shell=False,
            argv=argv,
            timeout_seconds=10 if mode == "cancel" else 2 if mode == "cpu" else 1,
        )
        native = {
            **old.native_manifest.model_dump(mode="json"),
            "entrypoint_uri": script.as_uri(),
            "resources": [resource],
            "invocation": invocation,
            "runtime_environment": {"admission_runtime": runtime},
        }
        manifest = reseal(old, native_manifest=native, invocation=invocation)
        profile = reseal(profile, manifest_checksum=manifest.checksum, runtime=runtime)
        probe = reseal(
            old_probe,
            manifest_checksum=manifest.checksum,
            profile_checksum=profile.checksum,
            runtime_checksum=digest(profile.runtime),
        )
        f.owners.manifests[
            (manifest.catalog_ref, manifest.tool_ref, manifest.implementation_ref)
        ] = manifest
        f.owners.profiles[manifest.checksum], f.owners.probes[profile.checksum] = profile, probe
        invocations.append(
            InvocationInput(
                implementation_ref=manifest.implementation_ref,
                manifest_checksum=manifest.checksum,
                profile_checksum=profile.checksum,
                parameters={},
            )
        )
        outputs = []
        for binding in profile.output_target_bindings:
            policy = PNGPolicy(
                artifact_ref=binding.constraint.artifact_ref,
                artifact_type_code=output_artifact_type
                or ("ARTIFACT:06D:" + digest(binding.constraint.artifact_ref)[7:39].upper()),
                unit="1" if binding.constraint.unit.state == "known" else None,
                unit_term=binding.constraint.unit,
                role="derived_product",
                labels=(0, 1),
                max_pixels=16777216,
            )
            qc_ref = ref("policy", "POLICY:06D:PNG_QC").model_copy(
                update={"checksum": digest(policy)}
            )
            policies[qc_ref] = policy
            outputs.append(
                OutputSpec(
                    port=binding.port,
                    filename=binding.port.port_name + ".png",
                    qc_policy_ref=qc_ref,
                )
            )
        reader.values[manifest.checksum] = seal(
            CommandAllowlist,
            version="1.0.0",
            source_ref=ref("source_version", "SOURCE:06D:FIXTURE_ALLOWLIST"),
            manifest_checksum=manifest.checksum,
            profile_checksum=profile.checksum,
            compiler_config_checksum=cfg.checksum,
            synthetic_only=True,
            executable_uri=executable.as_uri(),
            executable_checksum=checksum(executable.read_bytes()),
            entrypoint_uri=script.as_uri(),
            entrypoint_checksum=checksum(script.read_bytes()),
            resources=[resource],
            argv=argv,
            parameter_schema_version="1.0.0",
            parameter_schema_checksum=digest(manifest.parameter_schema),
            outputs=outputs,
        )
    # Reassemble the actual admission service; do not patch its verification methods.
    service = ToolAdmissionService(
        policy=AdmissionPolicy(max_total_bytes=268435456),
        grounding=f.grounding,
        catalogs=f.catalogs,
        manifests=f.owners,
        profiles=f.owners,
        probes=f.owners,
        inputs=f.owners,
        research=f.owners,
        objects=f.objects,
        resources=LocalResourcePolicy(allowed_roots=(str(root), str(executable.parent.parent))),
        clock=f.service.clock
        if hasattr(f.service, "clock")
        else __import__("tests.grounding.query_support", fromlist=["NOW"]).NOW,
    )
    return finish(f, service, reader, tuple(invocations), cfg, policies, root, repository, files)


def finish(f, service, reader, invocations, cfg, policies, root, repository, files):
    from tests.grounding.query_support import NOW

    service.clock = lambda: NOW
    selection = AdmissionSelection(
        candidate_refs=f.selection.candidate_refs, invocations=invocations
    )
    admission = service.admit(f.q, f.result, selection)
    assert admission.status == "preparation_only", admission.rejected
    context = FormalContext(
        request=f.q, query_result=f.result, selection=selection, admission=admission
    )
    compiler = ToolWorkflowCompiler(service, reader, cfg)
    plan = compiler.compile(context, choose(admission, prefer_producers=True))
    policy = ExecutionPolicy(
        reference=ref("policy", "POLICY:06D:OFFLINE_EXECUTION"),
        runtime_files=files or runtime_files(root),
        allowed_plan_checksums=(plan.checksum,),
        purpose=plan.document.purpose,
        synthetic_execution=True,
        cpu_seconds=1,
    )
    qc = TechnicalQCRegistry(policies)
    executor = ToolExecutionService(
        compiler=compiler,
        catalogs=f.catalogs,
        objects=f.objects,
        qc=qc,
        repository=repository,
        policy=policy,
        workspace=root / "attempts",
    )
    return SimpleNamespace(
        f=f,
        compiler=compiler,
        plan=plan,
        policy=policy,
        executor=executor,
        qc=qc,
        root=root,
        script=root / "harmless_06d.py",
    )
