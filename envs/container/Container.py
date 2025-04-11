import time
class Container():
    # IPS = ips requirement
	# RAM = ram requirement in MB
	# Size = container size in MB
    

    def __init__(self, ID, creationId, creationInterval, IPSModel, RAMModel, DiskModel, Environment, HostID=-1):
        self.id = ID
        self.creationID = creationId
        self.hostid = HostID
        self.env = Environment
        self.createAt = creationInterval
        self.startAt = self.env.interval
        self.totalExecTime = 0
        self.totalMigrationTime = 0
        self.lastMigrationTime = 0
        self.active = True
        self.destroyAt = -1
        self.lastContainerSize = 0

        self.ipsmodel = IPSModel
        self.ipsmodel.allocContainer(self)
        self.sla = self.ipsmodel.SLA
        self.rammodel = RAMModel
        self.rammodel.allocContainer(self)
        self.diskmodel = DiskModel
        self.diskmodel.allocContainer(self)

        self.waitingTime = 0

    def getTimeRemain(self):
        return self.sla - self.waitingTime - self.totalExecTime - self.totalMigrationTime

    def getWaitingtime(self):
        return self.waitingTime

    def getCPU(self):
        if self.hostid == -1:
            return 0
        ips = self.getApparentIPS()
        hostIpsCaps = self.getHost().ipsCapacity
        return min(100, 100 * (ips / hostIpsCaps))

    def getBaseIPS(self):
        return self.ipsmodel.getIPS()

    def getApparentIPS(self):
        
        if self.hostid == -1: 
            return self.ipsmodel.getMaxIPS()
        hostBaseIPS = self.getHost().getBaseIPS()
        hostIPSCap = self.getHost().ipsCapacity
        canUseIPS = (hostIPSCap - hostBaseIPS) / len(self.env.getContainersOfHost(self.hostid))
        if canUseIPS < 0:
            return 0
        r =  min(self.ipsmodel.getMaxIPS(), self.getBaseIPS() + canUseIPS)
        return r


    def getRAM(self):
        rsize, rread, rwrirte = self.rammodel.ram()
        self.lastContainerSize = rsize
        return rsize, rread, rwrirte
    
    def getDisk(self):
        return self.diskmodel.disk()
    
    def getContainerSize(self):
        if self.lastContainerSize == 0: 
            self.getRAM()
        return self.lastContainerSize
    
    def getHostID(self):
        return self.hostid
    
    def getHost(self):
        return self.env.getHostByID(self.hostid)

    def allocate(self, hostId, allocBw):
        lastMigrationTime = 0
        if self.hostid == -1:
            lastMigrationTime += self.getContainerSize() / allocBw
        elif self.hostid != hostId:
            lastMigrationTime += self.getContainerSize() / allocBw
            lastMigrationTime += abs(self.env.hostlist[self.hostid].latency - self.env.hostlist[hostId].latency)
        self.hostid = hostId
        self.totalMigrationTime += lastMigrationTime
        self.lastMigrationTime = lastMigrationTime
    
    def execute(self):
        assert self.hostid != -1
        execTime = self.env.intervaltime - self.lastMigrationTime
        apparentIPS = self.getApparentIPS()
        requiredExecTime = (self.ipsmodel.totalInstructions - self.ipsmodel.completedInstructions) / apparentIPS if apparentIPS else 0
        totalExeT = min(max(execTime, 0), requiredExecTime)
        execIPS = apparentIPS * totalExeT
        self.totalExecTime += totalExeT
        self.ipsmodel.completedInstructions += execIPS
        self.env.getHostByID(self.hostid).ips_used_in_step += execIPS
        self.lastMigrationTime = 0
        
        
    def allocateAndExecute(self, hostId, allocBw):
        self.allocate(hostId, allocBw)
        self.execute()

    def destroy(self):
        self.destroyAt = self.env.interval
        self.hostid = -1
        self.active = False

    def get_info(self):
        info = {
            "id": self.id,
            "created_at": self.createAt,
            "host_id": self.hostid,
            "ips": self.getBaseIPS(),
            "ram": self.getRAM(),
            "disk": self.getDisk(),
            "waiting_time": self.getWaitingtime(),
            "total_exe_time": self.totalExecTime,
            "total_migration_time": self.totalMigrationTime,
            "total_instruction": self.ipsmodel.totalInstructions,
            "completed_instruction": self.ipsmodel.completedInstructions,
        }

        return info