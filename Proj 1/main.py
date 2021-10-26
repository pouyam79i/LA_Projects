# Coded by Pouya Mohammadi - 9829039
# First LA project - Fall 2021

# Holds matrix info and applies the basics of row operations
class Matrix():

    # Initailize a matrix information from console
    def initialMatrix(self, row, col):
        self.row = int(row)
        self.col = int(col)
        self.matrix = []
        for i in range(row):
            subMat = []
            subMatRawString = str(input()).split()
            for j in range(col):
                # # print("Please enter entity of row: {}, col: {}".format(i, j))
                subMat.append(float(subMatRawString[j]))
            self.matrix.append(subMat)

    # Imports a matrix from console
    def setMatrix(self, matrix):
        if not matrix:
            # print("This is an empty matrix")
            return
        self.matrix = matrix
        self.row = len(matrix)
        self.col = len(matrix[0])
        # print("set vals:")
        # print(self.matrix)
        # print(self.row)
        # print(self.col)

    # changing ro algorithm
    def changeRow(self, srcRowIndex, decRowIndex):
        if srcRowIndex < 0 or srcRowIndex >= self.row:
            return False
        if decRowIndex < 0 or decRowIndex >= self.row:
            return False
        if srcRowIndex == decRowIndex:
            return True
        temp = self.matrix[srcRowIndex]
        self.matrix[srcRowIndex] = self.matrix[decRowIndex]
        self.matrix[decRowIndex] = temp
        # print("Row changed successfully--->")
        # print(self.matrix)
        return True

    # prints the matrix
    def printMatrix(self):
        for i in range(self.row):
            for j in range(self.col):
                print(self.matrix[i][j], end=' ')
            print()

    # Setters:
    def setRow(self, rowIndex, newRowValue):
        self.matrix[rowIndex] = newRowValue
    # Getters:
    def getEntity(self, rowIndex, colIndex):
        return self.matrix[rowIndex][colIndex]
    def getRow(self, rowIndex):
        return self.matrix[rowIndex]
    def getMatrix(self):
        return self.matrix
    def getRowCount(self):
        return self.row
    def getColCount(self):
        return self.col

# matrix solver - used for one matrix
class LAsolver():

    # Sets a raw matrix in LAsolver 
    def setRawMAtrix(self, rawMatrix):
        self.rawMatrix = Matrix()
        self.rawMatrix.setMatrix(rawMatrix)
        self.rref = 0
        self.XN = {}

    # Reset the solver memory
    def resetHistory(self):
        self.rawMatrix = 0
        self.rref = 0
        self.XN = {}

    # deploy solver algorithm in this function
    def produceRREF(self):
        if not self.rawMatrix:
            return False
        self.rref = Matrix()
        self.rref.setMatrix(self.rawMatrix.getMatrix())
        lead = 0
        for r in range(self.rref.getRowCount()):
            if lead >= self.rref.getColCount():
                return
            i = r
            while self.rref.getEntity(i, lead) == 0:
                i += 1
                if i == self.rref.getRowCount():
                    i = r
                    lead += 1
                    if self.rref.getColCount() == lead:
                        return
            self.rref.changeRow(i, r)
            # print("r: {}, lead: {}".format(r, lead))
            lv = self.rref.getEntity(r, lead)
            self.rref.setRow(r, [ mrx / float(lv) for mrx in self.rref.getRow(r)])
            for i in range(self.rref.getRowCount()):
                if i != r:
                    lv = self.rref.getEntity(i, lead)
                    self.rref.setRow(i, [ iv - lv*rv for rv,iv in zip(self.rref.getRow(r), self.rref.getRow(i))])
            lead += 1
        # print("RREF is formed: ")
        # self.rref.printMatrix()
        return True

    # It solves the Linear Algebra Equation - produceRREF() must be called be for that!
    def solveLAE(self):
        if self.rref == 0: return False
        for i in range(self.rref.getRowCount()):
            i_inverse = (self.rref.getRowCount() - 1) - i
            subtractor = 0
            for j in range(self.rref.getColCount()):
                if self.rref.getEntity(i_inverse, j) != 0:
                    n = j + 1
                    while n < (self.rref.getColCount() - 1):
                        if n in self.XN:
                            subtractor += self.XN[n] * self.rref.getEntity(i_inverse, n)
                        else:
                            subtractor += 10 * self.rref.getEntity(i_inverse, n)
                            self.XN.update({n: 10})
                        n = n + 1
                    self.XN.update({j: (self.rref.getEntity(i_inverse, self.rref.getColCount() - 1) - subtractor)})
                    break
        for j in range(self.rref.getColCount() - 1):
            if j in self.XN:
               pass
            else:
                self.XN.update({j: 10}) 
        return True
        # print("XN: ")
        # print(self.XN)

    # deploy result show algorithm
    def printResult(self):
        if self.rref == 0:
            print("Solver is given no matrix!")
            return
        # print("RREF is formed:")
        self.rref.printMatrix()
        # print("The LAE result:")
        for j in range(self.rref.getColCount() - 1):
            print("X{} = {}".format(j+1, self.XN[j]))

# main app sequence
def main():
    matrixInfo = (str(input()).split())
    # print(matrixInfo)
    rowNum = int(matrixInfo[0])
    colNum = int(matrixInfo[1])
    mtx = Matrix()
    mtx.initialMatrix(rowNum, colNum)
    # mtx.printMatrix()
    solver = LAsolver()
    solver.setRawMAtrix(mtx.getMatrix())
    solver.produceRREF()
    solver.solveLAE()
    solver.printResult()

# Runnning main function
if __name__=="__main__":
    main()
