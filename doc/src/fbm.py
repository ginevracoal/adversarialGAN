import numpy as np


def generate(H, timeSeries, offset_slope = (0, 0), amplitude = 1):
    """
    Genera un moto Browniano frattale in base alla serie temporale e all'esponente di
    Hurst passati come parametro, e' possibile specificare anche i coefficienti della retta.
    """
    X = timeSeries[:, np.newaxis]
    Y = timeSeries[np.newaxis, :]

    exp_2H = H * 2
    covar = np.abs(X) ** exp_2H + np.abs(Y) ** exp_2H - np.abs(X - Y) ** exp_2H
    covar /= 2

    covarCho = np.linalg.cholesky(covar)
    fbmDistr = covarCho.dot( np.random.randn( len(timeSeries) ).T )

    sysMatrix = np.vstack(( np.ones( len(timeSeries) ), timeSeries )).T
    return timeSeries, np.dot(sysMatrix, offset_slope) + amplitude * fbmDistr



class Prediction():
    """
    Classe per analizzare ed effettuare predizioni sul fBm.
    """
    
    def __init__(self, timeSeries, dataPoints):
        """
        Inizializza tutte le variabili che verranno usate dalla classe.
        Precalcola alcuni valori utilizzati per velocizzare l'esecuzione.
        Come parametri si aspetta i valori misurati sperimentalmente.
        """
        self._timeSeries = np.array(timeSeries)
        self._data = np.array(dataPoints)
        self._sysMatrix = np.vstack(( np.ones( len(self._timeSeries) ), self._timeSeries )).T

        self._defaultHRange = np.linspace(0.1,0.95, 100)

        self._estH = self.MLEstH()
        self._estOffsetSlope = self.MLEstOffsetSlope()
        
        self._predictionTimeSeries = None
        self._finalTimeSeries = None
        self._predSysMatrix = None
        self._A, self._B, self._C = None, None, None
        self._cov, self._mean = None, None
        self._covarCho = None


    def _covar(self, H, timeSeriesX, timeSeriesY = None):
        """
        Calcola la covarianza tra due serie temporali per il modello del fBm.
        Come parametri prende l'esponente di Hurst e una o due serie temporali, se la seconda non e' specificata assume i valori della prima.
        """
        if timeSeriesY == None:
            timeSeriesY = timeSeriesX

        ti = timeSeriesX[:, np.newaxis]
        tj = timeSeriesY[np.newaxis, :]

        exp_2H = H * 2
        cov = np.abs(ti) ** exp_2H + np.abs(tj) ** exp_2H - np.abs(ti - tj) ** exp_2H
        cov /= 2
        
        return cov


    def _computeCorrelation(self, H, timeSeriesX, timeSeriesY = None):
        """
        Calcola la covarianza tra due serie temporali per il modello studiato (fBm + retta).
        Come parametri prende l'esponente di Hurst e una o due serie temporali, se la seconda non e' specificata assume i valori della prima.
        """
        if timeSeriesY == None:
            timeSeriesY = timeSeriesX

        ti = timeSeriesX[:, np.newaxis]
        tj = timeSeriesY[np.newaxis, :]

        exp_2H = H * 2
        cov = np.abs(ti) ** exp_2H + np.abs(tj) ** exp_2H - np.abs(ti - tj) ** exp_2H
        cov /= 2
        
        return self._estOffsetSlope[0]**2 + (self._estOffsetSlope[1]**2) * ti * tj + cov


    def _betaStar(self, invCovMat):
        """
        Parte isolata della formula facente parte dell'inversione Bayesiana.
        A partire dall'inversa della matrice di covarianza, calcola i coefficienti della retta sulla quale va poi a innestarsi il fBm.
        """
        res = self._sysMatrix.T
        res = res.dot(invCovMat)
        res = res.dot(self._sysMatrix)
        
        res = np.linalg.inv(res)
        res = res.dot(self._sysMatrix.T)
        res = res.dot(invCovMat)
        res = res.dot(self._data)
        
        return res


    def _squaredR(self, invCovMat, betaStar):
        """
        Parte isolata della formula facente parte dell'inversione Bayesiana.
        Questa funzione e' stata isolata per comodita' e leggibilita'.
        """
        bread = self._data - self._sysMatrix.dot(betaStar)

        return (bread.T).dot(invCovMat).dot(bread)


    def _incorporateData(self, fbmDistr):
        """
        Funzione utilizzata per includere all'interno dei punti stimati, i valori dei punti misurati in partenza.
        """
        index_counter = 0
        for time in self._timeSeries:
            insertPosition = np.where(self._finalTimeSeries == time)[0]
            insertData = self._data[index_counter]
            fbmDistr = np.insert(fbmDistr, insertPosition, insertData)
            index_counter += 1

        return fbmDistr


    def posteriorHDistr(self, HSamples = None):
        """
        Funzione che si occupa tramite l'inversione Bayesiana della stima della distribuzione di probabilita' dell'esponente di Hurst a partire dai dati misurati sperimentalmente.
        Possiede un solo parametro opzionale che va a costituire lo spazio lineare all'interno del quale ricercare il valore dell'esponente.
        Nei calcoli viene usata la scala logaritmica per evitare che i valori manipolati vadano in overflow inficiando qualsiasi operazione.
        """
        if HSamples == None:
            HSamples = self._defaultHRange

        estimationArray = np.zeros(HSamples.size)
        for i in range(HSamples.size):
            HE = HSamples[i]
            covarianceMatrix = self._covar(HE, self._timeSeries)
            invCovarianceMatrix = np.linalg.inv(covarianceMatrix)
            
            beta = self._betaStar(invCovarianceMatrix)
            rSq = self._squaredR(invCovarianceMatrix, beta)

            res = np.log( np.linalg.det(covarianceMatrix) )
            res += np.log( np.linalg.det( (self._sysMatrix.T).dot(invCovarianceMatrix).dot(self._sysMatrix) ) )
            res += np.log(rSq) * len(self._timeSeries)
            res = 1 - 0.5 * res
            estimationArray[i] = res

        estimationArray -= np.amax(estimationArray)
        estimationArray = np.exp(estimationArray)
        
        return estimationArray


    def MLEstH(self, HSamples = None):
        """
        Ottiene il valore dell'esponente di Hurst calcolato tramite il metodo della massima verosimiglianza.
        """
        if HSamples == None:
            return self._estH;
        
        hDistr = self.posteriorHDistr(HSamples)
        return HSamples[ np.argmax(hDistr) ]


    def MLEstOffsetSlope(self, HSamples = None):
        """
        Calcola i valori dei coefficienti della retta a partire dall'esponente di Hurst calcolato tramite il metodo della massima verosimiglianza.
        """
        if HSamples == None:
            return self._estOffsetSlope

        HE = self.MLEstH(HSamples)
        invCovarianceMatrix = np.linalg.inv( self._covar(HE, self._timeSeries) )
        
        return self._betaStar(invCovarianceMatrix)


    def setPredictionTimeSeries(self, predictionTimeSeries):
        """
        Imposta la serie temporale per la quale effettuare le previsioni.
        Applica la regola del complemento di Schur per poter generare una struttura affine che contempli la nuova serie temporale.
        Precalcola i valori della media e della covarianza della matrice ottenuta.
        """
        self._predictionTimeSeries = np.setdiff1d(predictionTimeSeries, self._timeSeries, assume_unique = True)
        self._finalTimeSeries = np.union1d(self._timeSeries, self._predictionTimeSeries)
        
        self._predSysMatrix = np.vstack(( np.ones( len(self._predictionTimeSeries) ), self._predictionTimeSeries )).T

        self._A = self._computeCorrelation(self._estH, self._predictionTimeSeries)
        self._B = self._computeCorrelation(self._estH, self._predictionTimeSeries, self._timeSeries)
        self._C = self._computeCorrelation(self._estH, self._timeSeries)

        self._cov = self._A - self._B.dot( np.linalg.inv(self._C) ).dot(self._B.T)
        self._mean = self._B.dot( np.linalg.inv(self._C) ).dot(self._data)
        
        self._covarCho = np.linalg.cholesky(self._cov)

        return self._finalTimeSeries


    def getPrecision(self):
        """
        Funzione che restituisce tre curve che rappresentano la media e la precisione delle stime effettuate.
        """
        sigma = np.diagonal(self._cov)

        top = self._incorporateData(self._mean + sigma)
        mid = self._incorporateData(self._mean)
        bot = self._incorporateData(self._mean - sigma)

        return top, mid, bot


    def getInterpolation(self):
        """
        Genera una distribuzione a partire dalla covarianza ottenuta con l'applicazione della regola del complemento di Schur.
        """
        fbmDistr = self._covarCho.dot( np.random.randn( len(self._predictionTimeSeries) ).T ) + self._mean

        return self._incorporateData(fbmDistr)
