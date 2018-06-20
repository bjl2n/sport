# noinspection PyPep8Naming
class TMR:
    """Class to hold the information related to TMRs at a node, namely the node_name,
    submission time of the TMR, and occurrence time of the TMR (which can occur
    a variable amount of time after the submission).  Custom sorting of TMRs is
    done based on occurrence_time.
    @type node_name: str
    @param node_name: Name of the node where the TMR was initiated.
    @type submission_time: float
    @param submission_time: Time at which the TMR was submitted.
    @type occurrence_time: float
    @param occurrence_time: Time at which the TMR will occur or occurred.
    """
    def __init__(self, node_name, submission_time, occurrence_time):
        """Class to hold the information related to TMRs at a node, namely the node_name,
        submission time of the TMR, and occurrence time of the TMR (which can occur
        a variable amount of time after the submission).  Custom sorting of TMRs is
        done based on occurrence_time.
        @type node_name: str
        @param node_name: Name of the node where the TMR was initiated.
        @type submission_time: float
        @param submission_time: Time at which the TMR was submitted.
        @type occurrence_time: float
        @param occurrence_time: Time at which the TMR will occur or occurred.
        """
        self.node_name = node_name
        self.submission_time = submission_time
        self.occurrence_time = occurrence_time

    def __str__(self):
        """
        @rtype: str
        """
        TMR_str = 'TMR_' + self.node_name + '_' + str(self.submission_time) + \
                  '_' + str(self.occurrence_time) + '\n'
        return TMR_str

    def __eq__(self, other):
        """Two TMRs are equal if and only if all three attributes are the same.
        @type other: tmr.TMR
        """
        return (self.node_name == other.node_name) and \
            (self.submission_time == other.submission_time) and \
            (self.occurrence_time == other.occurrence_time)

    def __ne__(self, other):
        return not self.__eq__(other)

    # All other comparisons are based on the occurrence_time of each TMR.
    def __lt__(self, other):
        return (self.node_name == other.node_name) and \
            (self.occurrence_time < other.occurrence_time)

    def __le__(self, other):
        return (self.node_name == other.node_name) and \
            (self.occurrence_time <= other.occurrence_time)

    def __ge__(self, other):
        return (self.node_name == other.node_name) and \
            (self.occurrence_time >= other.occurrence_time)

    def __gt__(self, other):
        return (self.node_name == other.node_name) and \
            (self.occurrence_time > other.occurrence_time)
