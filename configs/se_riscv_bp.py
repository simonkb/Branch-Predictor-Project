# configs/se_riscv_bp.py
#
# Project-owned RISC-V SE runner with selectable branch predictors.
# This follows the *exact style/structure* of gem5's deprecated
# configs/deprecated/example/se.py, but extends --bp-type so you can select
# more predictors (BiModeBP, TournamentBP, LocalBP, TAGE, etc.) without
# modifying gem5 source code.
#
# Examples:
#   List predictors supported by THIS script:
#     ./gem5/build/RISCV/gem5.opt configs/se_riscv_bp.py --list-bp-types
#
#   Run:
#     ./gem5/build/RISCV/gem5.opt --outdir=results/hello_BiModeBP \
#       configs/se_riscv_bp.py \
#       --cmd=./benchmarks/hello/hello_rv_static \
#       --cpu-type=TimingSimpleCPU \
#       --bp-type=BiModeBP \
#       -I 1000000
#
# Notes:
# - Use static binaries in SE mode (your hello_rv_static is correct).
# - For GshareBP in gem5 25.1, conditionalBranchPred must be set; this script
#   does it automatically using LocalBP.

# Copyright (c) 2012-2013 ARM Limited
# All rights reserved.
#
# The license below extends only to copyright in the software and shall
# not be construed as granting a license to any other intellectual
# property including but not limited to intellectual property relating
# to a hardware implementation of the functionality of the software
# licensed hereunder.  You may use the software subject to the license
# terms below provided that you ensure that this notice is replicated
# unmodified and in its entirety in all distributions of the software,
# modified or unmodified, in source code or in binary form.
#
# Copyright (c) 2006-2008 The Regents of The University of Michigan
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Simple test script
#
# "m5 test.py"

import argparse
import os
import sys

import m5
from m5.defines import buildEnv
from m5.objects import *
from m5.params import NULL
from m5.util import (
    addToPath,
    fatal,
    warn,
)

from gem5.isas import ISA

import m5
# Ensure gem5 config "common" package is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEM5_CONFIGS = os.path.join(PROJECT_ROOT, "gem5", "configs")

if not os.path.isdir(GEM5_CONFIGS):
    fatal(f"Cannot find gem5 configs at {GEM5_CONFIGS}")

addToPath(GEM5_CONFIGS)
from common import (
    CacheConfig,
    CpuConfig,
    MemConfig,
    ObjectList,
    Options,
    Simulation,
)
from common.Caches import *
from common.cpu2000 import *
from common.FileSystemConfig import config_filesystem
from ruby import Ruby


def pop_arg(argv, name, default=None):
    """
    Remove an argument from argv if present.
    Supports:
      --name value
      --name=value
    Returns (value, new_argv)
    """
    out = [argv[0]]
    val = default
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == name:
            if i + 1 >= len(argv):
                fatal(f"{name} requires a value")
            val = argv[i + 1]
            i += 2
            continue
        if a.startswith(name + "="):
            val = a.split("=", 1)[1]
            i += 1
            continue
        out.append(a)
        i += 1
    return val, out

# ---- Extended Branch Predictor support (script-level) ------------------------

def _try_bp_cls(name: str):
    """Try to get BP SimObject class by name from m5.objects."""
    try:
        return getattr(__import__("m5.objects", fromlist=[name]), name)
    except Exception:
        pass

    try:
        mod = __import__("m5.objects.BranchPredictor", fromlist=[name])
        return getattr(mod, name)
    except Exception:
        return None


# These classes exist in gem5/src/cpu/pred/BranchPredictor.py + C++ side.
# We build our own list because deprecated se.py's ObjectList.bp_list is limited.
BP_CLASSES = {
    "LocalBP": _try_bp_cls("LocalBP"),
    "BiModeBP": _try_bp_cls("BiModeBP"),
    "TournamentBP": _try_bp_cls("TournamentBP"),
    "GshareBP": _try_bp_cls("GshareBP"),
    "LAPBP": _try_bp_cls("LAPBP"),
    "TAGE": _try_bp_cls("TAGE"),
    "LTAGE": _try_bp_cls("LTAGE"),
    "TAGE_SC_L": _try_bp_cls("TAGE_SC_L"),
    "TAGE_SC_L_8KB": _try_bp_cls("TAGE_SC_L_8KB"),
    "TAGE_SC_L_64KB": _try_bp_cls("TAGE_SC_L_64KB"),
}
# Filter out any missing classes (defensive)
BP_CLASSES = {k: v for k, v in BP_CLASSES.items() if v is not None}


def list_bp_types_and_exit():
    print("Available bp types (this script):")
    print("  LAP")
    for k in sorted(BP_CLASSES.keys()):
        print(f"  {k}")
    sys.exit(0)


def attach_branch_predictor(cpu, bp_name: str):
    """
    Attach branch predictor correctly in gem5 25.1.

    cpu.branchPred expects a BPredUnit. Many predictors (BiModeBP, TournamentBP,
    LocalBP, TAGE, LTAGE, etc.) are ConditionalPredictor objects and must be
    assigned under:
        cpu.branchPred.conditionalBranchPred
    """
    if not bp_name:
        return

    if bp_name not in BP_CLASSES and bp_name not in ("BranchPredictor", "LAP"):
        fatal(
            f"Unknown --bp-type '{bp_name}'. "
            "Use --list-bp-types to see available predictors."
        )

    # Case 1: user wants the default BPredUnit
    if bp_name == "BranchPredictor":
        bpu = BranchPredictor()
        # Make it functional by providing a default conditional predictor
        if "LocalBP" in BP_CLASSES:
            bpu.conditionalBranchPred = BP_CLASSES["LocalBP"]()
        cpu.branchPred = bpu
        return

    # Alias: LAP is a project-facing name that maps to gem5's LAPBP
    if bp_name == "LAP":
        if "LAPBP" not in BP_CLASSES:
            fatal("LAP selected but LAPBP was not found in this gem5 build.")
        cond = BP_CLASSES["LAPBP"]()
        bpu = BranchPredictor()
        bpu.conditionalBranchPred = cond
        cpu.branchPred = bpu
        return

    # Case 2: GshareBP is a BPredUnit, but needs conditionalBranchPred set in 25.1
    if bp_name == "GshareBP":
        bpu = BP_CLASSES["GshareBP"]()
        if "LocalBP" not in BP_CLASSES:
            fatal("GshareBP requires LocalBP but LocalBP was not found in this build.")
        bpu.conditionalBranchPred = BP_CLASSES["LocalBP"]()
        cpu.branchPred = bpu
        return

    # Case 3: Everything else here is a ConditionalPredictor -> wrap it
    cond = BP_CLASSES[bp_name]()  # e.g., BiModeBP(), TournamentBP(), TAGE(), ...
    bpu = BranchPredictor()
    bpu.conditionalBranchPred = cond
    cpu.branchPred = bpu
    
# ---- se.py behavior ---------------------------------------------------------

def get_processes(args):
    """Interprets provided args and returns a list of processes"""

    multiprocesses = []
    inputs = []
    outputs = []
    errouts = []
    pargs = []

    workloads = args.cmd.split(";")
    if args.input != "":
        inputs = args.input.split(";")
    if args.output != "":
        outputs = args.output.split(";")
    if args.errout != "":
        errouts = args.errout.split(";")
    if args.options != "":
        pargs = args.options.split(";")

    idx = 0
    for wrkld in workloads:
        process = Process(pid=100 + idx)
        process.executable = wrkld
        process.cwd = os.getcwd()
        process.gid = os.getgid()

        if args.env:
            with open(args.env) as f:
                process.env = [line.rstrip() for line in f]

        if len(pargs) > idx:
            process.cmd = [wrkld] + pargs[idx].split()
        else:
            process.cmd = [wrkld]

        if len(inputs) > idx:
            process.input = inputs[idx]
        if len(outputs) > idx:
            process.output = outputs[idx]
        if len(errouts) > idx:
            process.errout = errouts[idx]

        multiprocesses.append(process)
        idx += 1

    if args.smt:
        cpu_type = ObjectList.cpu_list.get(args.cpu_type)
        assert ObjectList.is_o3_cpu(cpu_type), "SMT requires an O3CPU"
        return multiprocesses, idx
    else:
        return multiprocesses, 1


warn(
    "Project script based on deprecated se.py style. "
    "This project-owned copy extends --bp-type support for branch predictors."
)
# Early exit for listing bp types without touching gem5 Options flags
if "--list-bp-types" in sys.argv:
    list_bp_types_and_exit()

# --- Override gem5's restricted --bp-type choices:
# Extract our desired bp type from argv, then remove it so gem5's Options parser
# doesn't reject it.
bp_type, new_argv = pop_arg(sys.argv, "--bp-type", default="BiModeBP")
sys.argv = new_argv

parser = argparse.ArgumentParser()
Options.addCommonOptions(parser)
Options.addSEOptions(parser)

if "--ruby" in sys.argv:
    Ruby.define_options(parser)

args = parser.parse_args()
args.bp_type = bp_type

multiprocesses = []
numThreads = 1

if args.bench:
    apps = args.bench.split("-")
    if len(apps) != args.num_cpus:
        print("number of benchmarks not equal to set num_cpus!")
        sys.exit(1)

    for app in apps:
        try:
            if ObjectList.cpu_list.get_isa(args.cpu_type) == ISA.ARM:
                exec(
                    "workload = %s('arm_%s', 'linux', '%s')"
                    % (app, args.arm_iset, args.spec_input)
                )
            else:
                # TARGET_ISA has been removed, but this is missing a ], so it
                # has incorrect syntax and wasn't being used anyway.
                exec(
                    "workload = %s(buildEnv['TARGET_ISA', 'linux', '%s')"
                    % (app, args.spec_input)
                )
            multiprocesses.append(workload.makeProcess())
        except:
            print(
                f"Unable to find workload for ISA: {app}",
                file=sys.stderr,
            )
            sys.exit(1)
elif args.cmd:
    multiprocesses, numThreads = get_processes(args)
else:
    print("No workload specified. Exiting!\n", file=sys.stderr)
    sys.exit(1)


(CPUClass, test_mem_mode, FutureClass) = Simulation.setCPUClass(args)
CPUClass.numThreads = numThreads

# Check -- do not allow SMT with multiple CPUs
if args.smt and args.num_cpus > 1:
    fatal("You cannot use SMT with multiple CPUs!")

np = args.num_cpus
mp0_path = multiprocesses[0].executable
system = System(
    cpu=[CPUClass(cpu_id=i) for i in range(np)],
    mem_mode=test_mem_mode,
    mem_ranges=[AddrRange(args.mem_size)],
    cache_line_size=args.cacheline_size,
)

if numThreads > 1:
    system.multi_thread = True

# Create a top-level voltage domain
system.voltage_domain = VoltageDomain(voltage=args.sys_voltage)

# Create a source clock for the system and set the clock period
system.clk_domain = SrcClockDomain(
    clock=args.sys_clock, voltage_domain=system.voltage_domain
)

# Create a CPU voltage domain
system.cpu_voltage_domain = VoltageDomain()

# Create a separate clock domain for the CPUs
system.cpu_clk_domain = SrcClockDomain(
    clock=args.cpu_clock, voltage_domain=system.cpu_voltage_domain
)

# If elastic tracing is enabled, then configure the cpu and attach the elastic
# trace probe
if args.elastic_trace_en:
    CpuConfig.config_etrace(CPUClass, system.cpu, args)

# All cpus belong to a common cpu_clk_domain, therefore running at a common
# frequency.
for cpu in system.cpu:
    cpu.clk_domain = system.cpu_clk_domain

if ObjectList.is_kvm_cpu(CPUClass) or ObjectList.is_kvm_cpu(FutureClass):
    if buildEnv["USE_X86_ISA"]:
        system.kvm_vm = KvmVM()
        system.m5ops_base = max(0xFFFF0000, Addr(args.mem_size).getValue())
        for process in multiprocesses:
            process.useArchPT = True
            process.kvmInSE = True
    else:
        fatal("KvmCPU can only be used in SE mode with x86")

# Sanity check
if args.simpoint_profile:
    if not ObjectList.is_noncaching_cpu(CPUClass):
        fatal("SimPoint/BPProbe should be done with an atomic cpu")
    if np > 1:
        fatal("SimPoint generation not supported with more than one CPUs")

for i in range(np):
    if args.smt:
        system.cpu[i].workload = multiprocesses
    elif len(multiprocesses) == 1:
        system.cpu[i].workload = multiprocesses[0]
    else:
        system.cpu[i].workload = multiprocesses[i]

    if args.simpoint_profile:
        system.cpu[i].addSimPointProbe(args.simpoint_interval)

    if args.checker:
        system.cpu[i].addCheckerCpu()

    # ---- Modified: use our extended BP_CLASSES instead of ObjectList.bp_list
    if args.bp_type:
        attach_branch_predictor(system.cpu[i], args.bp_type)

    if args.indirect_bp_type:
        indirectBPClass = ObjectList.indirect_bp_list.get(args.indirect_bp_type)
        system.cpu[i].branchPred.indirectBranchPred = indirectBPClass()

    system.cpu[i].createThreads()

if args.ruby:
    Ruby.create_system(args, False, system)
    assert args.num_cpus == len(system.ruby._cpu_ports)

    system.ruby.clk_domain = SrcClockDomain(
        clock=args.ruby_clock, voltage_domain=system.voltage_domain
    )
    for i in range(np):
        ruby_port = system.ruby._cpu_ports[i]

        # Create the interrupt controller and connect its ports to Ruby
        # Note that the interrupt controller is always present but only
        # in x86 does it have message ports that need to be connected
        system.cpu[i].createInterruptController()

        # Connect the cpu's cache ports to Ruby
        ruby_port.connectCpuPorts(system.cpu[i])
else:
    MemClass = Simulation.setMemClass(args)
    system.membus = SystemXBar()
    system.system_port = system.membus.cpu_side_ports
    CacheConfig.config_cache(args, system)
    MemConfig.config_mem(args, system)
    config_filesystem(system, args)

system.workload = SEWorkload.init_compatible(mp0_path)

if args.wait_gdb:
    system.workload.wait_for_remote_gdb = True

root = Root(full_system=False, system=system)
Simulation.run(args, root, system, FutureClass)