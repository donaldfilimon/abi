/**
 * Error handling for ABI JavaScript bindings.
 */

/**
 * Error codes returned by ABI operations.
 */
export enum ErrorCode {
  Ok = 0,
  InvalidArgument = 1,
  OutOfMemory = 2,
  NotInitialized = 3,
  AlreadyInitialized = 4,
  FeatureDisabled = 5,
  IoError = 6,
  NetworkError = 7,
  GpuError = 8,
  DatabaseError = 9,
  AgentError = 10,
  WasmError = 100,
  Unknown = 255,
}

/**
 * Error class for ABI operations.
 */
export class AbiError extends Error {
  /** Error code */
  public readonly code: ErrorCode;

  constructor(code: ErrorCode, message?: string) {
    const msg = message || AbiError.getDefaultMessage(code);
    super(msg);
    this.name = 'AbiError';
    this.code = code;
  }

  /**
   * Get default error message for a code.
   */
  static getDefaultMessage(code: ErrorCode): string {
    switch (code) {
      case ErrorCode.Ok:
        return 'Success';
      case ErrorCode.InvalidArgument:
        return 'Invalid argument';
      case ErrorCode.OutOfMemory:
        return 'Out of memory';
      case ErrorCode.NotInitialized:
        return 'Not initialized';
      case ErrorCode.AlreadyInitialized:
        return 'Already initialized';
      case ErrorCode.FeatureDisabled:
        return 'Feature disabled';
      case ErrorCode.IoError:
        return 'I/O error';
      case ErrorCode.NetworkError:
        return 'Network error';
      case ErrorCode.GpuError:
        return 'GPU error';
      case ErrorCode.DatabaseError:
        return 'Database error';
      case ErrorCode.AgentError:
        return 'Agent error';
      case ErrorCode.WasmError:
        return 'WASM error';
      default:
        return 'Unknown error';
    }
  }

  /**
   * Check if this is a specific error type.
   */
  isCode(code: ErrorCode): boolean {
    return this.code === code;
  }
}

/**
 * Check a result code and throw if error.
 */
export function checkError(code: number): void {
  if (code !== ErrorCode.Ok) {
    throw new AbiError(code as ErrorCode);
  }
}
