import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftGlobalEmptyArray(lldbtest.TestBase):
    @swiftTest
    def test(self):
        """Test that printing a global swift array of type SwiftEmptyArrayStorage uses the correct data formatter"""

        self.build()
        filespec = lldb.SBFileSpec("main.swift")
        lldbutil.run_to_source_breakpoint(
            self, "break here", filespec
        )
        x = self.frame().FindVariable("x")
        self.assertEqual(x.GetSummary(), "0 values")
